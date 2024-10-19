#!/bin/bash

# Create a backup directory
backup_dir="./backups/backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$backup_dir"

mv SCFout* SCFcalc* "$backup_dir"
mv *.out *.err *.log "$backup_dir"

echo "Backup created and files deleted successfully."