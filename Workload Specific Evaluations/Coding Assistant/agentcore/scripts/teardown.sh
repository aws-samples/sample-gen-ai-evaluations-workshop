#!/bin/bash
# Remove everything `p2_v2.py` created for one prefix, then check that nothing is left.
#   [PYTHON=.venv/bin/python] scripts/teardown.sh [PREFIX]   PREFIX defaults to $P2_PREFIX, then evals-p2b-
#   scripts/teardown.sh [PREFIX] --print  only print the commands of step 2 (for a shell where deletes are not allowed)
# Step 1 is `p2_v2.py cleanup`: AgentCore runtimes, ECR repository, CodeBuild project, three IAM roles.
# Step 2 is what cleanup does not do: the S3 source bucket and the log groups.
# Step 3 lists what is still there. Only names that start with PREFIX are touched.
set -u
export AWS_DEFAULT_REGION=us-east-1
HERE="$(cd "$(dirname "$0")/.." && pwd)"
PREFIX="${1:-${P2_PREFIX:-evals-p2b-}}"
case "$PREFIX" in evals-*) ;; *) echo "refusing prefix '$PREFIX': it must start with evals-"; exit 2;; esac
MODE="${2:-run}"
UPREFIX="${PREFIX//-/_}"                        # runtime names use underscores
ACCT=$(aws sts get-caller-identity --query Account --output text) || exit 1
BUCKET="${PREFIX}src-${ACCT}-us-east-1"

if [ "$MODE" != "--print" ]; then
  echo "### 1. python p2_v2.py --prefix $PREFIX cleanup"
  "${PYTHON:-python3}" "$HERE/p2_v2.py" --prefix "$PREFIX" cleanup
fi

CMDS=$(cat <<EOF
aws s3 rm s3://$BUCKET --recursive
aws s3api delete-bucket --bucket $BUCKET
aws logs delete-log-group --log-group-name /aws/codebuild/${PREFIX}build
for g in \$(aws logs describe-log-groups --log-group-name-prefix /aws/bedrock-agentcore/runtimes/${UPREFIX} --query 'logGroups[].logGroupName' --output text); do aws logs delete-log-group --log-group-name "\$g"; done
EOF
)
echo "### 2. bucket and log groups"
if [ "$MODE" = "--print" ]; then echo "$CMDS"; else
  # the AgentCore runtimes delete in the background: give them a moment so their log groups are final
  sleep 20
  eval "$CMDS"
fi

echo "### 3. what is left (every list below should be empty)"
echo "-- runtimes:";   "${PYTHON:-python3}" "$HERE/p2_v2.py" --prefix "$PREFIX" status 2>&1 | grep -v "None$" 
echo "-- iam roles:";  aws iam list-roles --query "Roles[?starts_with(RoleName, \`${PREFIX}\`)].RoleName" --output text
echo "-- ecr:";        aws ecr describe-repositories --query "repositories[?starts_with(repositoryName, \`${PREFIX}\`)].repositoryName" --output text 2>&1 | grep -v RepositoryNotFound
echo "-- codebuild:";  aws codebuild list-projects --query "projects[?starts_with(@, \`${PREFIX}\`)]" --output text
echo "-- s3:";         aws s3api list-buckets --query "Buckets[?starts_with(Name, \`${PREFIX}\`)].Name" --output text
echo "-- log groups:"; aws logs describe-log-groups --log-group-name-prefix "/aws/codebuild/${PREFIX}" --query 'logGroups[].logGroupName' --output text
aws logs describe-log-groups --log-group-name-prefix "/aws/bedrock-agentcore/runtimes/${UPREFIX}" --query 'logGroups[].logGroupName' --output text
echo "-- tagged Project=evals-streamline-spike (resources of other prefixes may show too):"
aws resourcegroupstaggingapi get-resources --tag-filters Key=Project,Values=evals-streamline-spike --query 'ResourceTagMappingList[].ResourceARN' --output text | tr '\t' '\n' | sed -E 's/[0-9]{12}/<ACCOUNT_ID>/g'
