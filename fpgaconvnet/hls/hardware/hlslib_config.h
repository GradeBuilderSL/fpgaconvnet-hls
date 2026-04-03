#pragma once

// Bridge Vitis HLS synthesis macro to hlslib's expected macro
#ifdef __SYNTHESIS__
#define HLSLIB_SYNTHESIS
#endif
