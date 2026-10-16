import os
import sys

# Windows only: faiss-cpu (libomp) and torch (libiomp5md) load two OpenMP runtimes into one process.
# Without this flag, the first faiss search after torch loads aborts with "OMP: Error #15".
# This is Intel's documented workaround; it is unsafe in general, so the vector-index tests also
# compare FAISS results against brute-force numpy search. Linux (Docker) does not need it.
if sys.platform == "win32":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
