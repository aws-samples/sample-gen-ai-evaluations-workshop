#!/bin/sh
# Start Squid with the initial policy from the environment: POLICY_MODE plus ALLOWED_HOSTS.
set -eu
# shellcheck disable=SC2086 - ALLOWED_HOSTS is a space-separated list, split on purpose.
policy "${POLICY_MODE:-deny-all}" ${ALLOWED_HOSTS:-}
touch /var/log/squid/access.log /var/log/squid/cache.log && chown -R squid:squid /var/log/squid /var/run/squid
tail -F /var/log/squid/access.log &
exec squid -N -f /etc/squid/squid.conf
