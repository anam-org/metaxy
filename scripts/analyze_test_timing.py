#!/usr/bin/env python3
"""Wrapper for mxtest-analyze-timings. Use the CLI directly instead:

mxtest-analyze-timings junit.xml
mxtest-analyze-timings junit.xml -f markdown
mxtest-analyze-timings junit.xml -f json
"""

from metaxy_testing.analyze_timings import main

if __name__ == "__main__":
    main()
