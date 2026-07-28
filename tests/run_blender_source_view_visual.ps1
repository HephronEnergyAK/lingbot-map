[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$BlenderPath,

    [Parameter(Mandatory = $true)]
    [string]$WorkerPython,

    [string]$RepositoryRoot = (Split-Path -Parent $PSScriptRoot),

    [string]$BlenderUserExtensions
)

$ErrorActionPreference = "Stop"
$repository = [System.IO.Path]::GetFullPath($RepositoryRoot)
$blender = [System.IO.Path]::GetFullPath($BlenderPath)
$python = [System.IO.Path]::GetFullPath($WorkerPython)
$temporaryRoot = [System.IO.Path]::GetFullPath(
    [System.IO.Path]::GetTempPath()
)
$runId = [System.Guid]::NewGuid().ToString("N").Substring(0, 12)
$fixture = [System.IO.Path]::GetFullPath(
    (Join-Path $temporaryRoot "lbmv-$runId")
)
$profile = [System.IO.Path]::GetFullPath(
    (Join-Path $temporaryRoot "lbmp-$runId")
)

foreach ($path in @($fixture, $profile)) {
    if (-not $path.StartsWith(
        $temporaryRoot,
        [System.StringComparison]::OrdinalIgnoreCase
    )) {
        throw "Refusing visual-test path outside the temporary root: $path"
    }
}
foreach ($executable in @($blender, $python)) {
    if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) {
        throw "Visual-test executable is absent: $executable"
    }
}

New-Item -ItemType Directory -Path $profile | Out-Null

& $python (
    Join-Path $repository "tests\build_source_view_fixture.py"
) $fixture
if ($LASTEXITCODE -ne 0) {
    throw "Source View fixture builder failed with exit code $LASTEXITCODE"
}

$previousConfig = $env:BLENDER_USER_CONFIG
$previousExtensions = $env:BLENDER_USER_EXTENSIONS
$previousInstalledGate = $env:LINGBOT_MAP_INSTALLED_EXTENSION
$previousProfileGate = $env:LINGBOT_MAP_VISUAL_ISOLATED_PROFILE
$previousFixture = $env:LINGBOT_MAP_SOURCE_VIEW_FIXTURE
try {
    $env:BLENDER_USER_CONFIG = $profile
    if ($BlenderUserExtensions) {
        $extensions = [System.IO.Path]::GetFullPath($BlenderUserExtensions)
        if (-not (Test-Path -LiteralPath $extensions -PathType Container)) {
            throw "Installed extension root is absent: $extensions"
        }
        $env:BLENDER_USER_EXTENSIONS = $extensions
        $env:LINGBOT_MAP_INSTALLED_EXTENSION = "1"
    } else {
        $env:LINGBOT_MAP_INSTALLED_EXTENSION = $null
    }
    $env:LINGBOT_MAP_VISUAL_ISOLATED_PROFILE = "1"
    $env:LINGBOT_MAP_SOURCE_VIEW_FIXTURE = $fixture
    & $blender --background --factory-startup --python (
        Join-Path $repository "tests\blender_visual_preferences.py"
    )
    if ($LASTEXITCODE -ne 0) {
        throw "Blender visual preference bootstrap failed with exit code $LASTEXITCODE"
    }
    if (-not (Test-Path -LiteralPath (Join-Path $profile "userpref.blend"))) {
        throw "Blender did not publish the isolated user preference file"
    }

    & $blender --python (
        Join-Path $repository "tests\blender_source_view_visual.py"
    )
    if ($LASTEXITCODE -ne 0) {
        throw "Blender visual oracle failed with exit code $LASTEXITCODE"
    }
} finally {
    $env:BLENDER_USER_CONFIG = $previousConfig
    $env:BLENDER_USER_EXTENSIONS = $previousExtensions
    $env:LINGBOT_MAP_INSTALLED_EXTENSION = $previousInstalledGate
    $env:LINGBOT_MAP_VISUAL_ISOLATED_PROFILE = $previousProfileGate
    $env:LINGBOT_MAP_SOURCE_VIEW_FIXTURE = $previousFixture
}

$markerPath = Join-Path $fixture "visual-marker.json"
if (-not (Test-Path -LiteralPath $markerPath -PathType Leaf)) {
    throw "Blender visual oracle did not publish its marker"
}
$marker = Get-Content -Raw -LiteralPath $markerPath | ConvertFrom-Json
if (
    $marker.state -ne "succeeded" -or
    $marker.blender_version -notlike "5.2*" -or
    @($marker.display_transforms).Count -ne 8 -or
    $marker.screenshots -ne 32 -or
    $marker.background_max_error_display_pixels -gt 1.0 -or
    $marker.coverage_max_error_display_pixels -gt 1.0 -or
    $marker.temporary_guide_datablocks -ne 0
) {
    throw "Blender visual oracle marker does not satisfy release bounds"
}

[pscustomobject]@{
    State = $marker.state
    BlenderVersion = $marker.blender_version
    DisplayTransforms = @($marker.display_transforms).Count
    Screenshots = $marker.screenshots
    BackgroundMaxErrorDisplayPixels = (
        $marker.background_max_error_display_pixels
    )
    CoverageMaxErrorDisplayPixels = (
        $marker.coverage_max_error_display_pixels
    )
    TemporaryGuideDatablocks = $marker.temporary_guide_datablocks
    InstalledExtension = $marker.installed_extension
    ExtensionModulePath = $marker.extension_module_path
} | ConvertTo-Json -Compress
