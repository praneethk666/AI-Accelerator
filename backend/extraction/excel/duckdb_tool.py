from typing import Any

import duckdb

from backend.extraction.excel.excel_tool import get_sheets, resolve_document_path


class DuckDBExcelTool:
    name = "duckdb_sql_query"
    description = (
        "Execute raw SQL queries directly over Excel and CSV spreadsheets.\n"
        "RULES:\n"
        "1. You must provide a valid SQLite/DuckDB SQL query as `sql_query`.\n"
        "2. The file must be specified by `filename_or_id`.\n"
        "3. A virtual table is created for the sheet, usually named after the sheet, or 'sheet1' if unnamed. To be safe, query `information_schema.tables` if unsure of the table name.\n"
    )
    from typing import ClassVar
    input_schema: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "filename_or_id": {"type": "string"},
            "sql_query": {"type": "string"}
        },
        "required": ["filename_or_id", "sql_query"]
    }

    def run(self, **kwargs) -> dict[str, Any]:
        filename_or_id = kwargs.get("filename_or_id") or kwargs.get("filename")
        sql_query = kwargs.get("sql_query")
        
        if not filename_or_id or not sql_query:
            return {"error": "filename_or_id and sql_query are required."}
            
        try:
            path = resolve_document_path(filename_or_id)
            sheets = get_sheets(path, sheet_name="all")
            
            # Create in-memory DuckDB connection
            con = duckdb.connect(database=':memory:')
            
            # Register all sheets as tables
            for sheet_name, df in sheets.items():
                # Clean sheet name for SQL safety (e.g. spaces to underscores)
                safe_name = "".join([c if c.isalnum() else "_" for c in sheet_name])
                # We register the dataframe so DuckDB can query it
                con.register(safe_name, df)
                # Also register as 'sheet1' if it's the first one, for convenience
                if list(sheets.keys()).index(sheet_name) == 0:
                    con.register('sheet1', df)

            # Execute query
            result_df = con.execute(sql_query).df()
            
            # Convert to markdown
            md_table = result_df.to_markdown(index=False)
            
            return {
                "blocks": [{"type": "table", "content": md_table}],
                "raw_result": result_df.to_dict(orient="records")
            }
        except Exception as e:
            return {"error": f"DuckDB SQL Execution failed: {e!s}"}
