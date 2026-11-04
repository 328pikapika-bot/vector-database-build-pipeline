param(
    [int]$StartPage = 10,
    [int]$Pages = 50,
    [string]$SubDocType = "A79",
    [int]$TimeoutSeconds = 900,
    [double]$IntervalSeconds = 2.0,
    [switch]$NoResume,
    [switch]$ContinueAfterTimeout
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $repoRoot

$pythonScript = Join-Path $repoRoot "vector_sync_service\scripts\publish_batch_pages.py"
$arguments = @(
    $pythonScript,
    "--start-page", $StartPage,
    "--pages", $Pages,
    "--sub-doc-type", $SubDocType,
    "--timeout-seconds", $TimeoutSeconds,
    "--interval-seconds", $IntervalSeconds
)

if ($NoResume) {
    $arguments += "--no-resume"
}

if ($ContinueAfterTimeout) {
    $arguments += "--continue-after-timeout"
}

python @arguments
exit $LASTEXITCODE
