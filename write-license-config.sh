#!/bin/sh

fname='changed_python_files.txt'
git diff --name-only --diff-filter=AMT origin/main| grep -E '\.(py|go|js|jsx|ts|tsx)$'|sed 's/^/    - /' > ${fname}
echo "Affected files in this PR that require copyright statement check:"
cat ${fname}
sed -i -e "/INSERT_FILES_FROM_PULL_REQUEST_HERE/r ${fname}" .licenserc.yaml
if [ -e .licenserc.yaml-e ]; then
  rm .licenserc.yaml-e
fi
rm ${fname}
