#!/usr/bin/env bash
# job_started_hook.sh -- the runners' ACTIONS_RUNNER_HOOK_JOB_STARTED (runner.sh
# copies it next to the runners and names it in their .env). It runs before every
# job on the runner; a non-zero exit fails the job before any of its steps.
#
# The repository is public, so a PR can add a workflow of its own that names this
# runner's label, and the runner would run it: a fork PR's `pull_request`
# workflow runs the PR's copy of the workflow files, and only GitHub's approval
# setting for outside contributors stands in the way (11-22, fork PRs on
# sgl-project/sglang mined on its GPU runners). A repository runner cannot be
# held to one workflow the way an organization runner group can, so this hook
# does it: only the PR test and the nightly, as main has them, on the events they
# run on, and a PR test only for a PR that has the run-ci or run-ci-extra label.
# A dispatch may come from any branch of the repository (to try a CI change before
# it merges): it needs write access, as pushing to main does.
#
# A PR cannot change what runs here: pull_request_target and schedule run main's
# workflow file, and a PR's own `pull_request` copy has the event and the ref
# (refs/pull/N/merge) refused below.
ALLOWED_REPO="sgl-project/sgl-amd-mono"
MAIN="refs/heads/main"
ALLOWED_WORKFLOWS="pr-test-amd.yml nightly-amd.yml"
# This copy's git blob hash, logged with every verdict: the copy on the runner is
# updated by hand (runner.sh hook), so compare it with
# `git ls-tree origin/main scripts/ci/job_started_hook.sh` to see that it is current.
version="$({ printf 'blob %d\0' "$(wc -c < "$0")"; cat "$0"; } | sha1sum | cut -c1-12)"
deny() {
  echo "::error::This runner only serves $ALLOWED_WORKFLOWS in $ALLOWED_REPO, from $MAIN but when dispatched (a PR needs the run-ci label). Refusing: $1 (hook $version)"
  exit 1
}
# GITHUB_WORKFLOW_REF = <owner>/<repo>/.github/workflows/<file>@<ref>
wf_ref="${GITHUB_WORKFLOW_REF:-}"
wf_path="${wf_ref%@*}"
wf_file="${wf_path##*/}"
wf_at="${wf_ref##*@}"
[[ "${GITHUB_REPOSITORY:-}" == "$ALLOWED_REPO" ]] || deny "repository '${GITHUB_REPOSITORY:-}'"
[[ "$wf_path" == "$ALLOWED_REPO/.github/workflows/$wf_file" ]] || deny "workflow '$wf_ref'"
[[ " $ALLOWED_WORKFLOWS " == *" $wf_file "* ]] || deny "workflow file '$wf_file'"
[[ "${GITHUB_REF:-}" == "$wf_at" ]] || deny "run ref '${GITHUB_REF:-}' (workflow from '$wf_at')"
case "$wf_file:${GITHUB_EVENT_NAME:-}" in
  pr-test-amd.yml:workflow_dispatch | nightly-amd.yml:workflow_dispatch)
    [[ "$wf_at" == refs/heads/* ]] || deny "workflow ref '$wf_at'"
    ;;
  nightly-amd.yml:schedule)
    [[ "$wf_at" == "$MAIN" ]] || deny "workflow ref '$wf_at'"
    ;;
  pr-test-amd.yml:pull_request_target)
    [[ "$wf_at" == "$MAIN" ]] || deny "workflow ref '$wf_at'"
    # The labels as of the event (the workflow's gate job reads them live, and
    # runs the test only with one). jq -e fails on a missing or unreadable payload.
    jq -e '[.pull_request.labels[]?.name] | any(. == "run-ci" or . == "run-ci-extra")' \
      "${GITHUB_EVENT_PATH:-/nonexistent}" > /dev/null 2>&1 \
      || deny "pull request without the run-ci label"
    ;;
  *) deny "event '${GITHUB_EVENT_NAME:-}' for $wf_file" ;;
esac
echo "job allowed: $wf_file@$wf_at (${GITHUB_EVENT_NAME}) run ${GITHUB_RUN_ID:-?} by ${GITHUB_TRIGGERING_ACTOR:-${GITHUB_ACTOR:-?}} (hook $version)"
