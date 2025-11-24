# Post-Incident Root Cause + Prevention Report

## 1. Root Cause Analysis
- Git kept trying to upload ~40,000 images because those files had been committed historically and remained in the repository’s packfiles. Even after they were deleted locally, their objects still existed in history and needed to be sent to the remote on push.
- The large folders (e.g., `data/raw`, `data/processed/preproc*`, `data/processed/embeddings*`) were tracked in the index and committed before deletion. Removing them from disk did not remove them from Git history, so pushes still contained those objects.
- `.gitignore` was incomplete; most large data directories and binary patterns were not ignored. This allowed processed and raw data to be added, staged, and committed.
- The Git index retained cached entries for deleted files, and the history retained the corresponding blobs. Untracked files don’t matter to pushes, but once staged/committed, their blobs remain in history until rewritten.
- Packfiles and stale objects from the old history still contained every image blob. During push, Git repacked/sent those objects, triggering the massive upload.

## 2. Why the problem escalated
- Git tried to resend all large files on every push because the remote lacked those objects; each push needed to deliver the entire set of missing blobs from history.
- The remote hung up during sideband packet transmission because the pack was large (2–3 GB), often exceeding network/time limits and causing timeouts.
- Large binary assets overwhelm Git’s diff/pack system: they don’t delta well, and thousands of blobs inflate pack size.
- `git count-objects -v` before cleanup showed tens of thousands of packed objects and multi-GB packfiles, confirming that garbage collection hadn’t removed the historical blobs.

## 3. Will this happen again?
- If nothing changes: High probability. Any future push would reintroduce large data unless history is filtered and ignores are correct.
- If `.gitignore` is correct: Medium–low. New large files won’t be added, but historical blobs remain unless already removed (they were removed via history rewrite).
- If cached objects are removed (history filtered): Low. The main risk is re-adding large files.
- If the repository is kept clean going forward: Very low. Remaining risk is accidental staging of large binaries.
- Risk factors: Adding/moving/renaming large folders (`data/raw`, `data/processed/preproc*`, embeddings, parquet/npy) without proper ignore rules; committing notebook outputs with embedded data.

## 4. Preventing recurrence
(A) Permanent `.gitignore` rules for large generated folders  
```
data/processed/preproc/
data/processed/preproc_backup_31396/
data/processed/preproc_best10k/
data/raw/keep_local_125000_files/
data/raw/for_review/
data/interim/
data/processed/embeddings/
*.npy
*.parquet
```
(B) Ensure Git does NOT track processed data: never run `git add` on these paths; keep them ignored; verify with `git status --ignored` to confirm they are excluded.  
(C) If it happens again: `git rm -r --cached <path>` for any data dirs that slipped in; commit the removals; if already in history, use `git filter-repo` or BFG to strip them, then force-push.  
(D) Run `git gc --prune=now` and `git repack -Ad` after removing tracked data or rewriting history; run `git gc` periodically to keep packfiles small.  
(E) Notebooks and large temps: avoid embedding large outputs; clear outputs for big data cells before commit; keep temporary exports in ignored paths.  
(F) Verify no large files are tracked before every commit: list staged files; scan for >50MB tracked objects; ensure `.gitignore` covers data paths.

## 5. Recommended verification commands
- Current index state: `git status -sb`  
- Tracked large objects: `git rev-list --objects --all | git cat-file --batch-check='%(objectname) %(objecttype) %(objectsize) %(rest)' | sort -k3 -n | tail`  
- Packfile sizes: `git count-objects -v`  
- Confirm `.gitignore` is applied: `git status --ignored` and inspect that data dirs show as ignored  
- Ensure only source/metadata is tracked: `git ls-tree -r --name-only HEAD | grep -E '^(data/|.*\\.(parquet|npy))'` (should return nothing if data is fully ignored)

## 6. Final Summary
Git was pushing ~40k image blobs because they were committed in earlier history and still lived in packfiles; `.gitignore` gaps let large data be tracked, and deletions didn’t remove historical objects. The push was huge because binary assets don’t delta well and inflated the pack to multiple gigabytes. We rewrote history to drop data directories, fixed `.gitignore`, and force-pushed a clean history. With ignores in place, index cleanup steps documented, and verification commands provided, reappearance risk is now very low. 
