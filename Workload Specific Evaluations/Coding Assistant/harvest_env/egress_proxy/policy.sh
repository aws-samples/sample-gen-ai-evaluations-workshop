#!/bin/sh
# Rewrite the proxy's policy: `policy allow <host>...`, `policy allow-all`, or `policy deny-all`.
# Hosts follow Harbor's allowed_hosts syntax; a leading "*." becomes Squid's ".domain" suffix match.
set -eu
mode="$1"; shift || true
{
  echo "http_port 3128"
  echo "acl CONNECT method CONNECT"
  echo "acl SSL_ports port 443"
  echo "http_access deny CONNECT !SSL_ports"
  case "$mode" in
    allow-all) echo "http_access allow all" ;;
    deny-all) echo "http_access deny all" ;;
    allow)
      if [ "$#" -gt 0 ]; then
        printf 'acl allowed_hosts dstdomain'
        for host in "$@"; do printf ' %s' "$(echo "$host" | sed 's/^\*\././')"; done
        echo
        echo "http_access allow allowed_hosts"
      fi
      echo "http_access deny all"
      ;;
    *) echo "unknown policy mode: $mode" >&2; exit 2 ;;
  esac
  echo "cache deny all"
  # Squid drops to its own user, which cannot open the container's stdio; log to files it owns. The
  # entrypoint tails the access log to stdout, so every host the agent asked for is in `docker logs`.
  echo "access_log /var/log/squid/access.log"
  echo "cache_log /var/log/squid/cache.log"
  echo "pid_filename /var/run/squid/squid.pid"
} > /etc/squid/squid.conf
if [ -f /var/run/squid/squid.pid ]; then squid -k reconfigure; fi
