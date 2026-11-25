#!/bin/bash
set -euo pipefail

weights_root=${1:-/remote-home/xufang/YGC/weights}
store=$weights_root/official/ripe
mkdir -p "$store"
exec 9>"$store/.download.lock"
flock -n 9

file=$store/ripe_weights.pth
if [[ -f $file ]]; then
    test "$(stat -c '%s' "$file")" = 59480614
    sha256sum "$file"
    exit 0
fi

host=cvg.hhi.fraunhofer.de
system_ca=/etc/ssl/certs/ca-certificates.crt
# The origin currently serves the RSA intermediate for an ECC leaf. Verify the
# matching HARICA intermediate against the system trust store for this request.
intermediate=$(curl -fsS --max-time 30 https://repo.harica.gr/certs/HARICA-GEANT-TLS-E1.der | openssl x509 -inform DER)
leaf=$(timeout 20 openssl s_client -connect "$host:443" -servername "$host" </dev/null 2>/dev/null | openssl x509)
openssl verify -CAfile "$system_ca" -verify_hostname "$host" -purpose sslserver \
    -untrusted <(printf '%s\n' "$intermediate") <(printf '%s\n' "$leaf")

pending=$(mktemp "$store/.ripe_weights.XXXXXX")
trap 'rm -f -- "$pending"' EXIT
curl -fsS --max-time 300 --cacert <(cat "$system_ca"; printf '%s\n' "$intermediate") \
    "https://$host/RIPE/ripe_weights.pth" -o "$pending"
test "$(stat -c '%s' "$pending")" = 59480614
mv -T -- "$pending" "$file"
sha256sum "$file"
