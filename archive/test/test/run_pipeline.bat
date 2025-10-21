@echo off
cd "c:\Users\Franz\OneDrive\01_Promotion\01 Data\new-pipeline\src\meningioma_ftir_pipeline\test"
matlab -batch "run('run_test.m')" > test_output.txt 2>&1
echo Pipeline execution completed. Check test_output.txt for results.