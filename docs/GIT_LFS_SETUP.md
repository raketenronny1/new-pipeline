# Git LFS Setup Guide

## Overview

This repository uses **Git Large File Storage (LFS)** to efficiently handle large data files (`.mat` files and PDFs) without bloating the repository size.

## What is Git LFS?

Git LFS replaces large files with small pointer files in your Git repository, while storing the actual file contents on a remote server. This keeps your repository lightweight while allowing you to version control large files.

## Files Tracked by LFS

The following file types are automatically tracked by Git LFS:

- **`.mat` files** - MATLAB data files (datasets, models, results)
- **`.pdf` files** - Documentation and tutorials

### Large Data Files (tracked via LFS)

```
data/
├── allspekTable.mat          (149 MB)
├── dataset_complete.mat      (794 MB)
├── data_table_train.mat      (359 MB)
├── data_table_test.mat       (166 MB)
├── metadata_all_patients.mat (<1 MB)
├── split_info.mat            (<1 MB)
└── wavenumbers.mat           (<1 MB)
```

## Setup for New Contributors

If you're cloning this repository for the first time:

### 1. Install Git LFS

**Windows (using Git for Windows):**
Git LFS is included with Git for Windows 2.x. Verify installation:
```powershell
git lfs version
```

**macOS (using Homebrew):**
```bash
brew install git-lfs
```

**Linux:**
```bash
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
```

### 2. Initialize Git LFS

In your local repository:
```bash
git lfs install
```

### 3. Clone the Repository

```bash
git clone https://github.com/raketenronny1/new-pipeline.git
cd new-pipeline
```

Git LFS will automatically download the large files.

## Verifying LFS Setup

Check which files are tracked by LFS:
```bash
git lfs ls-files
```

Check LFS status:
```bash
git lfs status
```

## Working with LFS Files

### Adding New Large Files

Just use normal git commands - files matching the patterns in `.gitattributes` will automatically be tracked by LFS:

```bash
git add data/new_dataset.mat
git commit -m "Add new dataset"
git push
```

### Checking File Size

To see which files are taking up space:
```bash
git lfs ls-files --size
```

### Pulling LFS Files

When pulling updates:
```bash
git pull
```

LFS files are automatically downloaded. If you want to pull only the pointers (not the actual files):
```bash
GIT_LFS_SKIP_SMUDGE=1 git pull
```

Then fetch specific files when needed:
```bash
git lfs pull
```

## Configuration

### .gitattributes

The `.gitattributes` file defines which files are tracked by LFS:

```
*.mat filter=lfs diff=lfs merge=lfs -text
*.pdf filter=lfs diff=lfs merge=lfs -text
```

### .gitignore

Results and temporary files are still ignored (not tracked):

```
results/eda/eda_results_PP1.mat
results/eda_pipeline/eda_results_PP1.mat
results/meningioma_ftir_pipeline/*.mat
models/meningioma_ftir_pipeline/*.mat
```

## Troubleshooting

### Files Not Downloaded

If LFS files appear as pointers instead of actual files:
```bash
git lfs pull
```

### Checking LFS Bandwidth

GitHub provides 1 GB of free LFS storage and 1 GB/month bandwidth per account. To check usage:
- Go to GitHub Settings → Billing → Git LFS Data

### Converting Existing Files to LFS

If you need to migrate existing large files already in git history:
```bash
git lfs migrate import --include="*.mat,*.pdf"
```

⚠️ **Warning**: This rewrites history. Coordinate with team before running.

## Best Practices

1. **Commit data files separately** from code changes for clearer history
2. **Use descriptive commit messages** when adding/updating data files
3. **Don't commit generated results** - keep results in `.gitignore`
4. **Check file sizes** before committing with `git lfs ls-files --size`
5. **Compress data when possible** before adding to repository

## GitHub LFS Limits

- **Free tier**: 1 GB storage, 1 GB/month bandwidth
- **Paid tier**: Additional packs available

For this repository:
- Current LFS usage: ~1.5 GB
- Consider archiving old datasets if approaching limits

## Resources

- [Git LFS Documentation](https://git-lfs.github.com/)
- [GitHub LFS Guide](https://docs.github.com/en/repositories/working-with-files/managing-large-files)
- [Atlassian Git LFS Tutorial](https://www.atlassian.com/git/tutorials/git-lfs)

---

**Setup Date**: October 28, 2025  
**LFS Version**: 3.6.1
