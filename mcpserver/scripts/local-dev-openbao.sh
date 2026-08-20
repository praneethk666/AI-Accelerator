#!/bin/bash
# Setup script for OpenBao SMTP policy and AppRole

set -e

echo "Creating read-only policy for MCP SMTP credentials..."
bao policy write mcp-smtp-readonly - <<EOF_POLICY
path "secret/data/mcp/smtp" {
  capabilities = ["read"]
}
EOF_POLICY

echo "Enabling AppRole auth method (if not already enabled)..."
bao auth enable approle || true

echo "Creating AppRole for MCP with the readonly policy..."
bao write auth/approle/role/mcp-agent \
    secret_id_ttl=0 \
    token_num_uses=0 \
    token_ttl=1h \
    token_max_ttl=24h \
    secret_id_num_uses=0 \
    policies="mcp-smtp-readonly"

echo "Fetching RoleID and SecretID..."
ROLE_ID=$(bao read -field=role_id auth/approle/role/mcp-agent/role-id)
SECRET_ID=$(bao write -f -field=secret_id auth/approle/role/mcp-agent/secret-id)

echo "================================================================"
echo "Setup Complete!"
echo "Credentials have been saved to .env.local."
echo "Please move or merge these contents into your .env file."
echo "Do not commit these files to version control."
echo "================================================================"

cat << EOF > .env.local
CREDENTIAL_PROVIDER=openbao
OPENBAO_URL=http://localhost:8200
OPENBAO_ROLE_ID=$ROLE_ID
OPENBAO_SECRET_ID=$SECRET_ID
EOF
