# Documentation Index

This folder contains all documentation for the Meningioma FTIR Classification Pipeline.

---

## 📖 Current Documentation (Active)

### **Start Here**
- **[REFACTORED_PIPELINE.md](REFACTORED_PIPELINE.md)** - **Current pipeline documentation**
  - How to use the refactored pipeline
  - Active file descriptions
  - Usage examples
  - Performance metrics

### **Development History**
- **[DEVELOPMENT_HISTORY.md](DEVELOPMENT_HISTORY.md)** - **Complete development history**
  - Timeline of all major changes
  - Evolution from Phase 1 → Phase 2 → Phase 3
  - Lessons learned
  - Archived file explanations

---

## 📚 Historical Documentation (Reference)

These documents describe previous versions of the pipeline and are kept for reference.

### Phase 2 Documentation (Patient-Wise CV - Original Implementation)
- **[PATIENT_WISE_CV_README.md](PATIENT_WISE_CV_README.md)** (294 lines)
  - Original patient-wise CV implementation
  - Now superseded by `run_patientwise_cv_direct.m`
  
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** (348 lines)
  - Phase 2 implementation details
  - Lists all Phase 2 files (now archived)
  
- **[COMPLETE_OVERVIEW.md](COMPLETE_OVERVIEW.md)** (477 lines)
  - Comprehensive Phase 2 overview
  - Detailed examples and workflows
  
- **[QUICK_START.md](QUICK_START.md)**
  - Quick start guide for Phase 2
  - Some instructions still applicable

### Integration Documentation
- **[integrated_workflow_with_qc.md](integrated_workflow_with_qc.md)** (1555 lines)
  - Quality control integration documentation
  - Detailed QC metrics and thresholds
  - Still relevant for understanding QC approach

### Cleanup Documentation
- **[CLEANUP_SUMMARY.md](CLEANUP_SUMMARY.md)** (129 lines)
  - First cleanup attempt summary
  - Documents transition from Phase 1 to Phase 2
  
- **[DIRECTORY_STRUCTURE.md](DIRECTORY_STRUCTURE.md)**
  - Historical directory structure
  - May be outdated

---

## 🗂️ Documentation Organization

```
docs/
├── README.md                           # This file - documentation index
│
├── REFACTORED_PIPELINE.md              # ⭐ CURRENT: Active pipeline docs
├── DEVELOPMENT_HISTORY.md              # ⭐ HISTORY: Complete timeline
│
└── [Historical Docs - Phase 1 & 2]
    ├── PATIENT_WISE_CV_README.md
    ├── IMPLEMENTATION_SUMMARY.md
    ├── COMPLETE_OVERVIEW.md
    ├── QUICK_START.md
    ├── integrated_workflow_with_qc.md
    ├── CLEANUP_SUMMARY.md
    └── DIRECTORY_STRUCTURE.md
```

---

## 🎯 Quick Navigation

### I want to...

**Use the current pipeline**
→ Read [REFACTORED_PIPELINE.md](REFACTORED_PIPELINE.md)

**Understand what changed and why**
→ Read [DEVELOPMENT_HISTORY.md](DEVELOPMENT_HISTORY.md)

**Learn about quality control**
→ Read [integrated_workflow_with_qc.md](integrated_workflow_with_qc.md)

**See detailed Phase 2 implementation**
→ Read [COMPLETE_OVERVIEW.md](COMPLETE_OVERVIEW.md)

**Understand patient-wise CV concept**
→ Read [PATIENT_WISE_CV_README.md](PATIENT_WISE_CV_README.md)

---

## 📝 Document Status Legend

- ✅ **Active** - Current documentation, actively maintained
- 📚 **Reference** - Historical documentation, kept for reference
- 🗄️ **Archived** - Outdated, superseded by newer docs

---

## Version History

| Date | Version | Description |
|------|---------|-------------|
| Oct 21, 2025 | 3.0 | Refactored pipeline - direct table access |
| Early 2025 | 2.0 | Patient-wise CV implementation |
| 2024 | 1.0 | Initial pipeline with averaging |

---

*Last Updated: October 21, 2025*
