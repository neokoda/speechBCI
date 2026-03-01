$dataDir = "data"
if (!(Test-Path -Path $dataDir)) {
    New-Item -ItemType Directory -Path $dataDir | Out-Null
    Write-Host "Created $dataDir directory."
}

$ProgressPreference = 'SilentlyContinue'

$files = @{
    "competitionData.tar.gz" = "https://datadryad.org/downloads/file_stream/2547369";
    "languageModel.tar.gz" = "https://datadryad.org/downloads/file_stream/2547356";
    "derived.tar.gz" = "https://datadryad.org/downloads/file_stream/2547370"
}

foreach ($file in $files.GetEnumerator()) {
    $outFile = Join-Path $dataDir $file.Key
    $url = $file.Value
    if (Test-Path $outFile) {
        Write-Host "$($file.Key) already exists. Skipping."
        continue
    }
    Write-Host "Downloading $($file.Key)..."
    try {
        Invoke-WebRequest -Uri $url -OutFile $outFile -UseBasicParsing
        Write-Host "Downloaded $($file.Key) successfully."
    } catch {
        Write-Error "Failed to download $($file.Key): $_"
    }
}
