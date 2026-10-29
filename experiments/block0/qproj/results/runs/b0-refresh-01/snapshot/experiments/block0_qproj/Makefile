# Put beside qproj_acc_test.c, qproj_data.h, and qproj_y_ref.h.
# Override the runtime location with: make CVW_COMMON=/path/to/common

.DEFAULT_GOAL := all
.DELETE_ON_ERROR:
SHELL := /bin/bash

HOST_CC ?= gcc
RISCV_PREFIX ?= riscv64-unknown-elf-
SPIKE ?= spike
ISA ?= rv64imc_zicsr_zifencei
CVW_ROOT ?= $(HOME)/Documents/ACompArc/cvw
CVW_COMMON ?= $(CVW_ROOT)/examples/C/common
# Existing rv64gc configuration: RTL smoke test, not the final RV64IMC hardware baseline.
RTL_CONFIG ?= rv64gc
BUILD_DIR ?= build/qproj

HOST_CFLAGS ?= -O2 -Wall -Wextra
RISCV_CFLAGS := -O2 -Wall -Wextra -march=$(ISA) -mabi=lp64 \
	-mcmodel=medany -fno-tree-loop-distribute-patterns
RISCV_LDFLAGS := -march=$(ISA) -mabi=lp64 -mcmodel=medany \
	-nostdlib -static -T $(CVW_COMMON)/test.ld

SOURCE := qproj_acc_test.c
HEADERS := qproj_data.h qproj_y_ref.h
HOST_BIN := $(BUILD_DIR)/qproj_acc_host
ELF := $(BUILD_DIR)/qproj_test.elf
OBJECTS := $(BUILD_DIR)/qproj_acc_test.o $(BUILD_DIR)/crt.o $(BUILD_DIR)/syscalls.o
TRACE := $(BUILD_DIR)/qproj_trace.log
DISASM := $(BUILD_DIR)/qproj_test.dis

.PHONY: all host riscv run-host run rtl test inspect trace disasm clean print-config

all: host riscv
host: $(HOST_BIN)
riscv: $(ELF)

$(BUILD_DIR):
	mkdir -p $@

$(HOST_BIN): $(SOURCE) $(HEADERS) | $(BUILD_DIR)
	$(HOST_CC) $(HOST_CFLAGS) $(SOURCE) -o $@

$(BUILD_DIR)/qproj_acc_test.o: $(SOURCE) $(HEADERS) | $(BUILD_DIR)
	$(RISCV_PREFIX)gcc $(RISCV_CFLAGS) -c $(SOURCE) -o $@

$(BUILD_DIR)/crt.o: $(CVW_COMMON)/crt.S | $(BUILD_DIR)
	$(RISCV_PREFIX)gcc $(RISCV_CFLAGS) -I $(CVW_COMMON) -c $< -o $@

$(BUILD_DIR)/syscalls.o: $(CVW_COMMON)/syscalls.c $(wildcard $(CVW_COMMON)/*.h) | $(BUILD_DIR)
	$(RISCV_PREFIX)gcc $(RISCV_CFLAGS) -I $(CVW_COMMON) -c $< -o $@

$(ELF): $(OBJECTS) $(CVW_COMMON)/test.ld
	$(RISCV_PREFIX)gcc $(RISCV_LDFLAGS) $(OBJECTS) -lgcc -o $@

run-host: $(HOST_BIN)
	"$(HOST_BIN)"

run: $(ELF)
	$(SPIKE) --isa=$(ISA) "$(ELF)"

# Keep setup and wsim in the same shell. Resolve the ELF before changing directory.
rtl: $(ELF)
	cd "$(CVW_ROOT)" && source ./setup.sh && \
		wsim --sim verilator "$(RTL_CONFIG)" --elf "$(abspath $(ELF))"

# A failing host test prevents the Spike test from running.
test: $(HOST_BIN) $(ELF)
	"$(HOST_BIN)"
	$(SPIKE) --isa=$(ISA) "$(ELF)"

inspect: $(ELF)
	$(RISCV_PREFIX)readelf -A $(ELF)
	$(RISCV_PREFIX)size $(ELF)
	$(RISCV_PREFIX)objdump -d --disassemble=qproj_accumulate $(ELF)

trace: $(ELF)
	$(SPIKE) -l --log=$(TRACE) --isa=$(ISA) "$(ELF)"

disasm: $(ELF)
	$(RISCV_PREFIX)objdump -d $(ELF) > $(DISASM)

# Remove only this Makefile's generated outputs; preserve source and headers.
clean:
	rm -f $(HOST_BIN) $(ELF) $(OBJECTS) $(TRACE) $(DISASM)

# The run recorder uses these resolved settings to capture the external runtime.
print-config:
	@printf '%s\n' "HOST_CC=$(HOST_CC)" "RISCV_PREFIX=$(RISCV_PREFIX)" \
		"SPIKE=$(SPIKE)" "ISA=$(ISA)" "CVW_ROOT=$(CVW_ROOT)" \
		"CVW_COMMON=$(CVW_COMMON)" "HOST_CFLAGS=$(HOST_CFLAGS)" \
		"RISCV_CFLAGS=$(RISCV_CFLAGS)" "RISCV_LDFLAGS=$(RISCV_LDFLAGS)" \
		"BUILD_DIR=$(BUILD_DIR)" "RTL_CONFIG=$(RTL_CONFIG)"
