##@ License and security tools

# Copyright Notices
# See .licenserc.yaml for configuration
# See http://dev-docs.drdev.io/developer_guide/workflows/copyright.html for more info
.PHONY: write-license-config
write-license-config:
	cp .licenserc.yaml .licenserc.yaml.temp
	./write-license-config.sh

.PHONY: restore-license-config
restore-license-config:
	mv .licenserc.yaml.temp .licenserc.yaml

.PHONY: fix-licenses
fix-licenses: write-license-config  ## Fix licenses (ensure config is preserved)
	$(MAKE) do-fix-licenses; \
	EXIT_CODE=$$?; \
	$(MAKE) restore-license-config; \
	exit $$EXIT_CODE

.PHONY: check-licenses
check-licenses: write-license-config  ## Check licenses (ensure config is preserved)
	$(MAKE) do-check-licenses; \
	EXIT_CODE=$$?; \
	$(MAKE) restore-license-config; \
	exit $$EXIT_CODE

.PHONY: do-fix-licenses
do-fix-licenses:  ## Fix licenses in repo
	docker run --rm -v $(CURDIR):/github/workspace ghcr.io/apache/skywalking-eyes/license-eye:eb0e0b091ea41213f712f622797e37526ca1e5d6 -v info -c .licenserc.yaml header fix

.PHONY: do-check-licenses
do-check-licenses:  ## Check licenses in repo
	docker run --rm -v $(CURDIR):/github/workspace ghcr.io/apache/skywalking-eyes/license-eye:eb0e0b091ea41213f712f622797e37526ca1e5d6 -v info -c .licenserc.yaml header check

.PHONY: check-licenses-ci
check-licenses-ci: write-license-config
	license-eye -v info -c .licenserc.yaml header check

# End of license-related targets
