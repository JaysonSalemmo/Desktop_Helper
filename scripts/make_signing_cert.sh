#!/bin/bash
# Create a stable self-signed code-signing certificate for Desktop Helper.
#
# WHY: an ad-hoc-signed .app gets a new code fingerprint on every rebuild, so
# macOS resets ALL its TCC permission grants (Microphone, Calendar, Screen
# Recording…) each time you rebuild. Signing with a STABLE identity instead
# means TCC keys the grant to the identity, and it survives rebuilds — grant
# once, never again. scripts/freeze_app.py auto-uses this cert when present.
#
# The cert is self-signed and untrusted (Gatekeeper still won't vouch for the
# app to OTHER machines — that needs a paid Developer ID), but codesign will
# sign with it and TCC honors the resulting stable identity locally. Run once.
#
#   bash scripts/make_signing_cert.sh
set -e

CERT_NAME="Desktop Helper Signing"
KEYCHAIN="$HOME/Library/Keychains/login.keychain-db"

if security find-certificate -c "$CERT_NAME" >/dev/null 2>&1; then
    echo "'$CERT_NAME' already exists in the keychain — nothing to do."
    exit 0
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

cat > "$TMP/cert.cnf" <<'EOF'
[req]
distinguished_name = dn
x509_extensions = v3
prompt = no
[dn]
CN = Desktop Helper Signing
[v3]
basicConstraints = critical,CA:FALSE
keyUsage = critical,digitalSignature
extendedKeyUsage = critical,codeSigning
EOF

openssl req -x509 -newkey rsa:2048 -sha256 -days 3650 -nodes \
    -keyout "$TMP/key.pem" -out "$TMP/cert.pem" -config "$TMP/cert.cnf"

# -legacy: OpenSSL 3's default PKCS12 MAC is unreadable by macOS's importer
openssl pkcs12 -export -legacy -out "$TMP/id.p12" \
    -inkey "$TMP/key.pem" -in "$TMP/cert.pem" -passout pass:dh

# -T /usr/bin/codesign -A: let codesign use the private key without prompting
security import "$TMP/id.p12" -k "$KEYCHAIN" -P dh -T /usr/bin/codesign -A

echo "Created '$CERT_NAME'. Rebuild with scripts/freeze_app.py to sign with it."
