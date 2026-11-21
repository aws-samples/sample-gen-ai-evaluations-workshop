
# ===== P2 v2 variants (coding tasks only, opt-in at run time: V-par, V-hard) =====
# A pristine, root-only copy of the sealed repository (base commit, as built above) for V-hard, and the variant scripts.
# Nothing here changes the default behaviour: tests/test.sh is still the one that runs unless the runner asks for a variant.
RUN cp -a /work/repo /opt/hidden/sealed-repo && chown -R root:root /opt/hidden/sealed-repo
COPY variants/ /opt/hidden/variants/
RUN chown -R root:root /opt/hidden && chmod -R go-rwx /opt/hidden && chmod 0700 /opt/hidden/variants/*.sh
