.DEFAULT_GOAL := all
.DELETE_ON_ERROR:
SHELL := /bin/bash

# Resolve paths relative to this Makefile.
TEST_DIR := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
EXPERIMENT_DIR := $(abspath $(TEST_DIR)/../../..)

HOST_CC ?= gcc
RISCV_PREFIX ?= riscv64-unknown-elf-
SPIKE ?= spike
ISA ?= rv64imc_zicsr

CVW_ROOT ?= $(HOME)/Documents/ACompArc/cvw
CVW_COMMON ?= $(CVW_ROOT)/examples/C/common

BUILD_DIR ?= $(EXPERIMENT_DIR)/build/block0/profile

HOST_CFLAGS ?= -std=c11 -O2 -Wall -Wextra -Werror
RISCV_CFLAGS := -std=gnu11 -O2 -Wall -Wextra \
	-march=$(ISA) -mabi=lp64 -mcmodel=medany \
	-fno-tree-loop-distribute-patterns

RISCV_LDFLAGS := -march=$(ISA) -mabi=lp64 -mcmodel=medany \
	-nostdlib -static -T "$(CVW_COMMON)/test.ld"

SOURCES := $(TEST_DIR)/test_block0_profile.c \
	$(EXPERIMENT_DIR)/src/attention.c \
	$(EXPERIMENT_DIR)/src/gelu.c \
	$(EXPERIMENT_DIR)/src/layernorm.c \
	$(EXPERIMENT_DIR)/src/requantization.c \
	$(EXPERIMENT_DIR)/src/projection.c \
	$(EXPERIMENT_DIR)/src/residual.c \
	$(EXPERIMENT_DIR)/src/blocks/block0.c \
	$(EXPERIMENT_DIR)/params/block0_params.c \
	$(EXPERIMENT_DIR)/fixtures/block0/mlp_out_residual_golden.c

HEADERS := $(EXPERIMENT_DIR)/src/attention.h \
	$(EXPERIMENT_DIR)/src/gelu.h \
	$(EXPERIMENT_DIR)/src/layernorm.h \
	$(EXPERIMENT_DIR)/src/requantization.h \
	$(EXPERIMENT_DIR)/src/projection.h \
	$(EXPERIMENT_DIR)/src/residual.h \
	$(EXPERIMENT_DIR)/src/blocks/block0.h \
	$(EXPERIMENT_DIR)/params/block0_params.h \
	$(EXPERIMENT_DIR)/fixtures/block0/mlp_out_residual_golden.h

HOST_BIN := $(BUILD_DIR)/test_block0_profile_host
ELF := $(BUILD_DIR)/test_block0_profile.elf
DISASM := $(BUILD_DIR)/test_block0_profile.dis

RISCV_OBJECTS := $(patsubst $(EXPERIMENT_DIR)/%.c,$(BUILD_DIR)/riscv/%.o,$(SOURCES))
RUNTIME_OBJECTS := $(BUILD_DIR)/riscv/runtime/crt.o \
	$(BUILD_DIR)/riscv/runtime/syscalls.o
OBJECTS := $(RISCV_OBJECTS) $(RUNTIME_OBJECTS)
DEPS := $(OBJECTS:.o=.d)

.PHONY: all host riscv run-host run test inspect disasm clean

all: host riscv

host: $(HOST_BIN)

riscv: $(ELF)

$(HOST_BIN): $(SOURCES) $(HEADERS)
	mkdir -p "$(@D)"
	$(HOST_CC) $(HOST_CFLAGS) $(SOURCES) -o "$@"

# Dependency files track headers used by each RISC-V source.
$(BUILD_DIR)/riscv/%.o: $(EXPERIMENT_DIR)/%.c
	mkdir -p "$(@D)"
	$(RISCV_PREFIX)gcc $(RISCV_CFLAGS) -MMD -MP -c "$<" -o "$@"

$(BUILD_DIR)/riscv/runtime/crt.o: $(CVW_COMMON)/crt.S
	mkdir -p "$(@D)"
	$(RISCV_PREFIX)gcc $(RISCV_CFLAGS) -MMD -MP \
		-I "$(CVW_COMMON)" -c "$<" -o "$@"

$(BUILD_DIR)/riscv/runtime/syscalls.o: $(CVW_COMMON)/syscalls.c
	mkdir -p "$(@D)"
	$(RISCV_PREFIX)gcc $(RISCV_CFLAGS) -MMD -MP \
		-I "$(CVW_COMMON)" -c "$<" -o "$@"

$(ELF): $(OBJECTS) $(CVW_COMMON)/test.ld
	$(RISCV_PREFIX)gcc $(RISCV_LDFLAGS) $(OBJECTS) -lgcc -o "$@"

run-host: $(HOST_BIN)
	"$(HOST_BIN)"

run: $(ELF)
	$(SPIKE) --isa=$(ISA) "$(ELF)"

# Run sequentially: a failed host test prevents the Spike run.
test: $(HOST_BIN) $(ELF)
	"$(HOST_BIN)"
	$(SPIKE) --isa=$(ISA) "$(ELF)"

inspect: $(ELF)
	$(RISCV_PREFIX)readelf -A "$(ELF)"
	$(RISCV_PREFIX)size "$(ELF)"

disasm: $(DISASM)

$(DISASM): $(ELF)
	$(RISCV_PREFIX)objdump -d "$<" > "$@"

# Remove only this test's generated files.
clean:
	rm -f "$(HOST_BIN)" "$(ELF)" "$(DISASM)" $(OBJECTS) $(DEPS)

-include $(DEPS)