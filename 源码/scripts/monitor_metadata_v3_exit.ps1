param(
    [Parameter(Mandatory = $true)]
    [int]$WorkerProcessId,
    [ValidateRange(5, 300)]
    [int]$IntervalSeconds = 30
)

$ErrorActionPreference = 'Stop'
$stateDirectory = Join-Path (Split-Path -Parent $PSScriptRoot) 'out\metadata_v3_2024_2026'
$progressPath = Join-Path $stateDirectory 'progress.json'
$outputPath = Join-Path $stateDirectory 'exit_observer_20261026.jsonl'

function Write-DiagnosticRecord {
    param([string]$EventName, [hashtable]$Details)
    $record = [ordered]@{
        at = (Get-Date).ToString('yyyy-MM-ddTHH:mm:ssK')
        event = $EventName
        worker_pid = $WorkerProcessId
    }
    foreach ($key in $Details.Keys) { $record[$key] = $Details[$key] }
    $line = $record | ConvertTo-Json -Depth 5 -Compress
    [System.IO.File]::AppendAllText($outputPath, $line + [Environment]::NewLine, [System.Text.Encoding]::UTF8)
}

function Get-DiagnosticSnapshot {
    param([System.Diagnostics.Process]$Worker)
    $snapshot = @{}
    try {
        # Allow the worker to atomically replace progress.json while we read it.
        $share = [System.IO.FileShare]::ReadWrite -bor [System.IO.FileShare]::Delete
        $stream = [System.IO.File]::Open($progressPath, [System.IO.FileMode]::Open,
            [System.IO.FileAccess]::Read, $share)
        try {
            $reader = [System.IO.StreamReader]::new($stream, [System.Text.Encoding]::UTF8, $true)
            try { $progress = $reader.ReadToEnd() | ConvertFrom-Json }
            finally { $reader.Dispose() }
        } finally { $stream.Dispose() }
        $snapshot.progress_state = $progress.state
        $snapshot.progress_at = $progress.updated_at
        $snapshot.processed = $progress.processed
        $snapshot.failed = $progress.failed
        $snapshot.current_year = $progress.current_year
    } catch { $snapshot.progress_read_error = $_.Exception.GetType().Name }
    try {
        $Worker.Refresh()
        if (-not $Worker.HasExited) {
            $snapshot.cpu_seconds = [math]::Round($Worker.TotalProcessorTime.TotalSeconds, 1)
            $snapshot.working_set_mb = [math]::Round($Worker.WorkingSet64 / 1MB, 1)
            $snapshot.private_mb = [math]::Round($Worker.PrivateMemorySize64 / 1MB, 1)
        }
    } catch { $snapshot.process_read_error = $_.Exception.GetType().Name }
    try {
        $os = Get-CimInstance Win32_OperatingSystem
        $snapshot.free_memory_mb = [math]::Round($os.FreePhysicalMemory / 1024)
    } catch { $snapshot.memory_read_error = $_.Exception.GetType().Name }
    try {
        $snapshot.d_free_gb = [math]::Round((Get-PSDrive -Name D).Free / 1GB, 2)
    } catch { $snapshot.disk_read_error = $_.Exception.GetType().Name }
    return $snapshot
}

try {
    $worker = [System.Diagnostics.Process]::GetProcessById($WorkerProcessId)
    $workerStart = $worker.StartTime.ToString('yyyy-MM-ddTHH:mm:ssK')
} catch {
    Write-DiagnosticRecord 'observer_attach_failed' @{error_type = $_.Exception.GetType().Name}
    exit 1
}

Write-DiagnosticRecord 'observer_started' (@{worker_started_at = $workerStart} + (Get-DiagnosticSnapshot $worker))
while ($true) {
    if ($worker.WaitForExit($IntervalSeconds * 1000)) {
        $code = $worker.ExitCode
        $details = Get-DiagnosticSnapshot $worker
        $details.exit_code = $code
        $details.exit_code_hex = ('0x{0:X8}' -f [uint32]([int64]$code -band 0xFFFFFFFF))
        $details.worker_started_at = $workerStart
        try {
            $since = (Get-Date).AddMinutes(-2)
            $events = foreach ($logName in @('Application', 'System')) {
                Get-WinEvent -FilterHashtable @{LogName = $logName; StartTime = $since; Id = 1000, 1001, 1002, 41, 6008} -ErrorAction SilentlyContinue |
                    Select-Object @{Name = 'log'; Expression = {$logName}},
                        @{Name = 'at'; Expression = {$_.TimeCreated.ToString('yyyy-MM-ddTHH:mm:ssK')}},
                        Id, ProviderName
            }
            $details.recent_os_events = @($events)
        } catch { $details.event_log_read_error = $_.Exception.GetType().Name }
        Write-DiagnosticRecord 'worker_exited' $details
        break
    }
    Write-DiagnosticRecord 'heartbeat' (Get-DiagnosticSnapshot $worker)
}
