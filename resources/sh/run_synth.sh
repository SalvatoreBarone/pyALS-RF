#!/bin/bash
$1 -mode batch -nojournal -nolog -notrace -source create_project.tcl | tee synt_log.txt
