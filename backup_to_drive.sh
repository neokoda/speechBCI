#!/usr/bin/env bash
# =============================================================================
# backup_to_drive.sh — End-of-session backup to Google Drive via rclone
# =============================================================================
# Backs up crucial untracked research artifacts to gdrive:speechBCI_backup/
#
# Targets (~141 GB total):
#   experiments/      8.9 GB  — model checkpoints + eval results
#   data/derived/     9.8 GB  — TFRecords + baseline RNN checkpoint
#   speech_5gram/   122   GB  — 5-gram WFST (critical, hours to rebuild)
#   docs/             ~5  MB  — PDFs + HANDOFF.md
#
# Usage:
#   bash backup_to_drive.sh               # full backup
#   bash backup_to_drive.sh --dry-run     # preview without uploading
#   bash backup_to_drive.sh --skip-lm    # skip the 122 GB speech_5gram/
# =============================================================================

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
RCLONE="/usr/bin/rclone"
REMOTE_NAME="gdrive"
REMOTE="gdrive:speechBCI_backup"
REPO="/workspace/speechBCI"
LOG="/tmp/backup_$(date +%Y%m%d_%H%M%S).log"
DRY_RUN=false
SKIP_LM=false

# ── Parse flags ───────────────────────────────────────────────────────────────
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --skip-lm) SKIP_LM=true ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

# ── Preflight checks ──────────────────────────────────────────────────────────
echo "============================================================"
echo "  SpeechBCI → Google Drive Backup"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Log: $LOG"
if $DRY_RUN; then echo "  [DRY RUN — nothing will be uploaded]"; fi
if $SKIP_LM; then echo "  [--skip-lm: speech_5gram/ will be skipped]"; fi
echo "============================================================"
echo ""

if [[ ! -x "$RCLONE" ]]; then
    echo "ERROR: rclone not found at $RCLONE" >&2
    exit 1
fi

if ! "$RCLONE" listremotes 2>/dev/null | grep -q "^${REMOTE_NAME}:"; then
    echo "ERROR: rclone remote '${REMOTE_NAME}' is not configured." >&2
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  One-time setup (run once per new instance):"
    echo ""
    echo "  1. Start rclone config:"
    echo "       /usr/bin/rclone config"
    echo ""
    echo "  2. At the prompts:"
    echo "       n          → new remote"
    echo "       gdrive     → name (must be exactly 'gdrive')"
    echo "       drive      → type (Google Drive)"
    echo "       [Enter]    → client_id (leave blank)"
    echo "       [Enter]    → client_secret (leave blank)"
    echo "       1          → scope (full access)"
    echo "       [Enter]    → root_folder_id (leave blank)"
    echo "       n          → edit advanced config"
    echo "       n          → use auto config (headless server)"
    echo ""
    echo "  3. rclone will print a URL — open it on your LOCAL machine,"
    echo "     authenticate with your Google account, then paste the"
    echo "     verification code back into this terminal."
    echo ""
    echo "  4. Configure as Shared Drive? n"
    echo "     Confirm: y"
    echo ""
    echo "  Then re-run this script."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    exit 1
fi

# ── rclone flags ──────────────────────────────────────────────────────────────
# --checksum          Compare MD5 checksums instead of mtime+size. Critical for
#                     vast.ai: each new instance resets file mtimes, so without
#                     --checksum rclone would re-upload every unchanged file.
#
# --drive-chunk-size 128M  Upload in 128 MB chunks (default: 8 MB). Fewer API
#                     calls per GB → avoids hitting Google Drive's per-second
#                     and per-100-second request quota limits.
#
# --tpslimit 5        Cap at 5 API transactions/second. Keeps the upload safely
#                     under Google's rate limits for sustained runs.
#
# --transfers 4       4 concurrent file transfers for throughput without
#                     multiplying API call rate.
#
# --retries 10        Retry failed transfers up to 10× with exponential back-off.
# --retries-sleep 30s 30-second pause between retries (lets quota refill).
#
# --stats 30s         Print transfer stats every 30 seconds (visible in log).

RCLONE_FLAGS=(
    --checksum
    --drive-chunk-size 128M
    --tpslimit 5
    --transfers 4
    --retries 10
    --retries-sleep 30s
    --progress
    --stats 30s
    --log-file "$LOG"
    --log-level INFO
)

if $DRY_RUN; then
    RCLONE_FLAGS+=(--dry-run)
fi

# ── Helper ────────────────────────────────────────────────────────────────────
run_backup() {
    local label="$1"
    local src="$2"
    local dest="$3"
    shift 3
    local extra=("$@")

    echo "------------------------------------------------------------"
    echo "  $label"
    echo "  src : $src"
    echo "  dest: $dest"
    echo "------------------------------------------------------------"

    if [[ ! -e "$src" ]]; then
        echo "  WARNING: source path not found — skipping"
        echo ""
        return 0
    fi

    local t0
    t0=$(date +%s)

    if "$RCLONE" copy "${RCLONE_FLAGS[@]}" "${extra[@]}" "$src" "$dest"; then
        echo "  Done in $(( $(date +%s) - t0 ))s"
    else
        echo "  ERROR: rclone exited non-zero — check $LOG" >&2
        # non-fatal: continue with remaining targets
    fi
    echo ""
}

# ── Backup targets ────────────────────────────────────────────────────────────

# 1. experiments/ — 8.9 GB
run_backup \
    "experiments/ (checkpoints + eval results)" \
    "$REPO/experiments" \
    "$REMOTE/experiments" \
    --exclude "__pycache__/**" \
    --exclude "*.pyc"

# 2. data/derived/ — 9.8 GB
run_backup \
    "data/derived/ (TFRecords + baseline RNN)" \
    "$REPO/data/derived" \
    "$REMOTE/data/derived"

# 3. speech_5gram/ — only G.fst (5 GB) and words.txt
# G_no_prune.fst (75 GB) and TLG.fst (41 GB) are excluded: they are only needed
# for lattice rescoring which requires 128+ GB RAM and is currently blocked.
# They can be rebuilt on a high-RAM instance when needed.
if $SKIP_LM; then
    echo "------------------------------------------------------------"
    echo "  speech_5gram/ — SKIPPED (--skip-lm)"
    echo "------------------------------------------------------------"
    echo ""
else
    run_backup \
        "speech_5gram/ (G.fst + words.txt only; ~5 GB)" \
        "$REPO/speech_5gram" \
        "$REMOTE/speech_5gram" \
        --include "G.fst" \
        --include "words.txt" \
        --exclude "*"
fi

# 4. Root-level research docs — ~5 MB
echo "------------------------------------------------------------"
echo "  docs/ (PDFs + HANDOFF.md)"
echo "  dest: $REMOTE/docs"
echo "------------------------------------------------------------"

ROOT_DOCS=(
    "HANDOFF.md"
    "s41586-023-06377-x.pdf"
    "13522108-ProposalTA-signed.pdf"
    "laporanTugasAkhir-13521081-FINALFINAL.docx.pdf"
)

t0=$(date +%s)
docs_ok=true
for doc in "${ROOT_DOCS[@]}"; do
    src="$REPO/$doc"
    if [[ ! -f "$src" ]]; then
        echo "  WARNING: not found — skipping: $doc"
        continue
    fi
    echo "  Uploading: $doc"
    if ! "$RCLONE" copy "${RCLONE_FLAGS[@]}" "$src" "$REMOTE/docs/"; then
        echo "  ERROR uploading $doc" >&2
        docs_ok=false
    fi
done
if $docs_ok; then
    echo "  Done in $(( $(date +%s) - t0 ))s"
else
    echo "  Completed with errors — check $LOG" >&2
fi
echo ""

# ── Summary ───────────────────────────────────────────────────────────────────
echo "============================================================"
echo "  Backup complete: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "  Destination : $REMOTE/"
echo "  Layout:"
echo "    speechBCI_backup/"
echo "    ├── experiments/     (checkpoints + eval results)"
echo "    ├── data/derived/    (TFRecords + baseline RNN)"
if ! $SKIP_LM; then
echo "    ├── speech_5gram/    (5-gram WFST)"
fi
echo "    └── docs/            (PDFs + HANDOFF.md)"
echo ""
echo "  Full log    : $LOG"
if $DRY_RUN; then
    echo ""
    echo "  [DRY RUN — nothing was actually uploaded]"
fi
echo "============================================================"
