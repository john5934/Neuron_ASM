;-------------------------------------------------------------------------------
; platform_nasm.inc
;-------------------------------------------------------------------------------
;
; PURPOSE: Platform-specific definitions for the NASM assembler.
;
; DESCRIPTION: Include this in NASM source files first to set defaults for the
; current platform.
;
; HISTORY:
;     7Jun2020
;-----------------------------------------------------------------------------*/

cpu x64 ; Use x86-64 instructions set (x64/amd64/Intel 64).

; Use RIP-relative addressing by default on 64-bit Linux platforms to support
; position independence.
default rel

%stacksize flat64

section .text
