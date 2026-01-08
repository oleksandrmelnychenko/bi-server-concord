# Export full ConcordDb_v5 schema metadata (JSON) and table data (BCP native format)
# Uses sqlcmd + bcp available on Windows SQL Server client tools.
# Credentials default to repo values; override with env vars as needed.

param(
    [string]$Server = $(if ($env:MSSQL_HOST) { $env:MSSQL_HOST } else { "78.152.175.67" }),
    [int]$Port = $(if ($env:MSSQL_PORT) { [int]$env:MSSQL_PORT } else { 1433 }),
    [string]$Database = $(if ($env:MSSQL_DATABASE) { $env:MSSQL_DATABASE } else { "ConcordDb_v5" }),
    [string]$User = $(if ($env:MSSQL_USER) { $env:MSSQL_USER } else { "ef_migrator" }),
    [string]$Password = $(if ($env:MSSQL_PASSWORD) { $env:MSSQL_PASSWORD } else { "Grimm_jow92" }),
    [string]$OutDir = "db-dump"
)

Write-Host "Server: $Server`nDatabase: $Database`nOutDir: $OutDir"

# Create output folders
$schemaDir = Join-Path $OutDir "schema"
$dataDir = Join-Path $OutDir "data"
New-Item -ItemType Directory -Force -Path $schemaDir | Out-Null
New-Item -ItemType Directory -Force -Path $dataDir | Out-Null

# 1) Export schema metadata to JSON using existing extractor (requires python + pymssql)
$python = Get-Command python -ErrorAction SilentlyContinue
if ($python) {
    Write-Host "Extracting schema metadata to $schemaDir\schema_cache.json..."
    Push-Location db-ai-api
    try {
        $logPath = Join-Path -Path ".." -ChildPath (Join-Path $schemaDir "schema_extract.log")
        $schemaOut = Join-Path -Path ".." -ChildPath (Join-Path $schemaDir "schema_cache.json")
        python extract_full_schema.py | Tee-Object -FilePath $logPath
        Copy-Item schema_cache.json $schemaOut -Force
    } catch {
        Write-Warning "Schema extraction failed: $_"
    } finally {
        Pop-Location
    }
} else {
    Write-Warning "Python not found; skipping schema JSON extract."
}

# 2) Get table list (user tables only)
$env:SQLCMDPASSWORD = $Password
$tableList = sqlcmd -S "$Server,$Port" -U $User -d $Database -h -1 -W -Q "
SET NOCOUNT ON;
SELECT QUOTENAME(TABLE_SCHEMA) + '.' + QUOTENAME(TABLE_NAME)
FROM INFORMATION_SCHEMA.TABLES
WHERE TABLE_TYPE = 'BASE TABLE'
AND TABLE_SCHEMA NOT IN ('sys','INFORMATION_SCHEMA')
ORDER BY 1;
"

if (-not $tableList) {
    Write-Warning "No tables returned; verify credentials/network."
    exit 1
}

Write-Host "Exporting $($tableList.Count) tables via bcp (native format)..."

foreach ($t in $tableList) {
    $table = $t.Trim()
    if (-not $table) { continue }

    # File-safe name: schema.table.bcp
    $safeName = $table.Replace('[','').Replace(']','').Replace('.','_')
    $outFile = Join-Path $dataDir "$safeName.bcp"

    $query = "SELECT * FROM $table"
    $bcpCmd = @(
        "bcp",
        "`"$query`"",
        "queryout",
        "`"$outFile`"",
        "-S", "$Server,$Port",
        "-d", $Database,
        "-U", $User,
        "-P", $Password,
        "-n",              # native format
        "-b", "50000",     # batch size
        "-q"               # quoted identifiers
    )

    Write-Host "  -> $table => $outFile"
    $proc = Start-Process -FilePath $bcpCmd[0] -ArgumentList $bcpCmd[1..$bcpCmd.Length] -Wait -PassThru -NoNewWindow
    if ($proc.ExitCode -ne 0) {
        Write-Warning "BCP failed for $table (exit $($proc.ExitCode))"
    }
}

Write-Host "Export complete. Data files in $dataDir, schema JSON in $schemaDir."
