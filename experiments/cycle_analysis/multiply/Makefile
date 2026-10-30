.DEFAULT_GOAL := riscv
SHELL := /bin/bash

EXPERIMENT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
PROJECT_DIR := $(abspath $(EXPERIMENT_DIR)/../../..)
CVW_ROOT ?= $(HOME)/Documents/ACompArc/cvw
COMMON_DIR := $(CVW_ROOT)/examples/C/common
BUILD_DIR ?= $(PROJECT_DIR)/build/cycle_analysis/multiply

CROSS_COMPILE ?= riscv64-unknown-elf-
CC := $(CROSS_COMPILE)gcc
OBJDUMP := $(CROSS_COMPILE)objdump
WALLY_CONFIG ?= rv64gc
SIM ?= verilator

ELF := $(BUILD_DIR)/test_multiply.elf
DIS := $(BUILD_DIR)/test_multiply.dis
FUNCTION := multiply_i8
SOURCES := $(addprefix $(EXPERIMENT_DIR)/,main.c multiply.c)
RUNTIME_SOURCES := $(COMMON_DIR)/crt.S $(COMMON_DIR)/syscalls.c
DATA_DIR := 
DATA_HEADER := 
INCLUDES := -I "$(COMMON_DIR)" $(if $(DATA_DIR),-I "$(DATA_DIR)")
HEADERS := $(wildcard $(EXPERIMENT_DIR)/*.h) $(wildcard $(COMMON_DIR)/*.h)
HEADERS += $(if $(DATA_HEADER),$(DATA_DIR)/$(DATA_HEADER))

CFLAGS := -std=gnu11 -O2 -Wall -Wextra \
          -march=rv64imc_zicsr -mabi=lp64 -mcmodel=medany \
          -fno-tree-loop-distribute-patterns
LDFLAGS := -nostdlib -static -T "$(COMMON_DIR)/test.ld"

.PHONY: riscv run disasm inspect clean
riscv: $(ELF)

$(ELF): $(SOURCES) $(RUNTIME_SOURCES) $(HEADERS) $(COMMON_DIR)/test.ld $(EXPERIMENT_DIR)/Makefile
	mkdir -p "$(@D)"
	$(CC) $(CFLAGS) $(INCLUDES) $(foreach file,$(SOURCES) $(RUNTIME_SOURCES),"$(file)") $(LDFLAGS) -lgcc -o "$@"

run: $(ELF)
	cd "$(CVW_ROOT)" && source ./setup.sh && \
	    wsim --sim "$(SIM)" "$(WALLY_CONFIG)" --elf "$(abspath $(ELF))"

disasm: $(DIS)

$(DIS): $(ELF)
	$(OBJDUMP) -d "$<" > "$@"

inspect: $(ELF)
	$(OBJDUMP) -d --disassemble=$(FUNCTION) "$<"

# Keep generated input/reference headers when cleaning binaries.
clean:
	rm -f "$(ELF)" "$(DIS)"
