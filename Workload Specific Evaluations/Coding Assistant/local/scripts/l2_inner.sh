#!/bin/bash
# L2, runs INSIDE the agent's srt profile. Prints one line per gate: GATE|name|expected|observed|detail
g() { echo "GATE|$1|$2|$3|$4"; }
http() { curl -sS -o /dev/null -w '%{http_code}' --max-time 12 "$@" 2>&1 | tr '\n' ' ' | cut -c1-160; }
# classification: only a real HTTP status from the far side is REACHED. Any curl error (7 connect failed, 6 resolve, 56 proxy refused CONNECT, 28 timeout) is blocked.
# A bare 403 is the proxy's own refusal of a plain-HTTP request (no far-side server answered).
blocked() { case "$1" in *"curl: ("*) echo blocked;; 403) echo "blocked (proxy 403)";; [1-5][0-9][0-9]) echo REACHED;; *) echo blocked;; esac; }
for h in bedrock-mantle.us-east-1.api.aws bedrock-runtime.us-east-1.amazonaws.com; do
  o=$(http "https://$h/"); case "$o" in [1-5][0-9][0-9]) v="reachable (HTTP $o)";; *) v="NOT reachable: $o";; esac
  g "allowed-host $h" "reachable, HTTP status" "$v" ""
done
for h in github.com pypi.org registry.npmjs.org files.pythonhosted.org example.com; do
  o=$(http "https://$h/"); g "proxy->$h" blocked "$(blocked "$o")" "$o"
done
o=$(http "http://example.com/"); g "proxy http://example.com:80" blocked "$(blocked "$o")" "$o"
o=$(http --noproxy '*' "https://github.com/"); g "direct (no proxy) github.com" blocked "$(blocked "$o")" "$o"
o=$(http --noproxy '*' "https://1.1.1.1/"); g "direct 1.1.1.1:443" blocked "$(blocked "$o")" "$o"
o=$(http "https://1.1.1.1/"); g "proxy->1.1.1.1" blocked "$(blocked "$o")" "$o"
o=$(http -6 --noproxy '*' "https://[2606:4700:4700::1111]/"); g "IPv6 direct 2606:4700:4700::1111" blocked "$(blocked "$o")" "$o"
o=$(http --noproxy '*' "https://bedrock-runtime.us-east-1.amazonaws.com/"); g "direct (no proxy) allowed host" blocked "$(blocked "$o")" "$o"
( unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy; o=$(http "https://github.com/"); g "proxy vars unset, github.com" blocked "$(blocked "$o")" "$o" )
# (a bash-builtin UDP probe was removed: it printed a path instead of a result; the python UDP probe below covers UDP)
if [ -n "$P1_PY" ]; then
  r=$("$P1_PY" - <<'PY' 2>&1 | tail -1
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.settimeout(4)
try:
    s.sendto(b"\x12\x34\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00\x06github\x03com\x00\x00\x01\x00\x01", ("1.1.1.1", 53)); s.recvfrom(512); print("REACHED: got a DNS reply")
except Exception as e: print("blocked:", repr(e)[:100])
PY
); g "UDP DNS query to 1.1.1.1 (python)" blocked "$r" ""
  r=$("$P1_PY" - <<'PY' 2>&1 | tail -1
import socket
try: print("REACHED", socket.gethostbyname("github.com"))
except Exception as e: print("blocked:", repr(e)[:100])
PY
); g "local DNS lookup github.com (getaddrinfo)" blocked "$r" ""
  r=$("$P1_PY" - <<'PY' 2>&1 | tail -1
import socket
for host, port in (("127.0.0.1", 22), ("127.0.0.1", 5000), ("93.184.216.34", 80)):
    s = socket.socket(); s.settimeout(3)
    try: s.connect((host, port)); print("REACHED", host, port); break
    except Exception as e: pass
else: print("blocked: 127.0.0.1:22, 127.0.0.1:5000, 93.184.216.34:80")
PY
); g "raw TCP connect to other ports/hosts" blocked "$r" ""
fi
r=$(nc -z -w 3 127.0.0.1 22 2>&1 | tail -1 | cut -c1-100); g "nc 127.0.0.1:22" blocked "${r:-no connection}" ""
g "proxy variables present inside sandbox" ">0" "$(env | grep -ci proxy)" ""
