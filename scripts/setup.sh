#!/bin/bash
python -m scripts.init_db
python -m scripts.test_db
python -m scripts.import_placements
python -m scripts.check_placements

