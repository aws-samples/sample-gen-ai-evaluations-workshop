#!/bin/bash
# Root-owned bootstrap, run once at container start by /opt/shim/main.py (as root).
# The microVM gives root no CAP_NET_ADMIN, so nft/iptables cannot load.
# Design instead: the agent runs in an EMPTY network namespace (only lo) that is owned by a user namespace
# (so root can bring lo up without CAP_NET_ADMIN). The only way out is a relay on lo:3128 in that netns,
# which forwards to a unix socket served by the allowlist proxy (user `proxy`) in the main netns.
set -u
PORT=3128
D=/run/fw
LOG=/var/log/fw-start.log; : >"$LOG"; chmod 0600 "$LOG"
exec > >(tee -a "$LOG") 2>&1
fail() { echo "FW-FAIL: $*"; echo "FAIL $*" >/run/fw.status; exit 1; }
rm -f /run/fw.status

install -d -m 0755 -o root -g root $D
# 1. allowlist proxy in the main netns, user `proxy`, unix socket only
install -o proxy -g proxy -m 0640 /dev/null /var/log/proxy.log
install -d -m 0700 -o proxy -g proxy $D/sock
setsid setpriv --reuid=proxy --regid=proxy --clear-groups python3 /opt/firewall/proxy.py serve $D/sock/proxy.sock >>/var/log/proxy.log 2>&1 </dev/null &
for i in $(seq 1 50); do [ -S $D/sock/proxy.sock ] && break; sleep 0.1; done
[ -S $D/sock/proxy.sock ] || fail "proxy socket missing"

# 2. empty netns owned by a fresh userns; keep it alive with a bind mount
setsid unshare -Urn bash -c "ip link set lo up && echo \$\$ >$D/holder.pid && exec sleep infinity" >/dev/null 2>&1 </dev/null &
for i in $(seq 1 50); do [ -s $D/holder.pid ] && break; sleep 0.1; done
[ -s $D/holder.pid ] || fail "netns holder did not start"
: >$D/agent.net; chmod 0600 $D/agent.net
mount --bind /proc/$(cat $D/holder.pid)/ns/net $D/agent.net || fail "bind netns"

# 3. relay inside that netns (user proxy), lo:3128 -> unix socket
setsid nsenter --net=$D/agent.net setpriv --reuid=proxy --regid=proxy --clear-groups \
  python3 /opt/firewall/proxy.py relay $PORT $D/sock/proxy.sock >>/var/log/proxy.log 2>&1 </dev/null &
for i in $(seq 1 50); do nsenter --net=$D/agent.net ss -ltn "sport = :$PORT" | grep -q LISTEN && break; sleep 0.1; done
nsenter --net=$D/agent.net ss -ltn "sport = :$PORT" | grep -q LISTEN || fail "relay not listening"

install -m 0644 -o root -g root /dev/null $D/allow.txt   # empty = deny all until the runner sets a phase
install -m 0600 -o root -g root /dev/null $D/extra.env   # per-phase env for as-agent (scoped credentials)
# 4. hardening (best effort): no new user namespaces
sysctl -qw user.max_user_namespaces=0 2>&1 || echo "WARN: could not disable userns"
echo OK >/run/fw.status
echo "FW-OK"
