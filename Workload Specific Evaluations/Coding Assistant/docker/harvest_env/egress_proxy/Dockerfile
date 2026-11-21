# The allowlisting egress proxy for harvest_env.docker_allowlist. Squid, because CONNECT-by-hostname
# allowlisting is exactly its acl model; Alpine, because the image is tiny and built once per host.
FROM public.ecr.aws/docker/library/alpine:3.20
RUN apk add --no-cache squid && mkdir -p /var/log/squid /var/run/squid && chown -R squid:squid /var/log/squid /var/run/squid
COPY --chmod=755 policy.sh /usr/local/bin/policy
COPY --chmod=755 entrypoint.sh /usr/local/bin/entrypoint
EXPOSE 3128
ENTRYPOINT ["/usr/local/bin/entrypoint"]
