.DEFAULT_GOAL := riscv
.DELETE_ON_ERROR:
SHELL := /bin/bash

TEST_DIR := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
PROJECT_DIR := $(abspath $(TEST_DIR)/../../..)
SW_TESTS_DIR := $(PROJECT_DIR)/experiments/component_tests
SW_DIR := $(PROJECT_DIR)/src/sw
PARAMS_DIR ?= $(PROJECT_DIR)/build/parameter_export/c
GOLDEN_DIR ?= $(PROJECT_DIR)/build/fixtures/ln2
BUILD_DIR ?= $(PROJECT_DIR)/build/tests/ln2

RISCV_PREFIX ?= riscv64-unknown-elf-
SPIKE ?= spike
ISA ?= rv64imc_zicsr
CVW_ROOT ?= $(HOME)/Documents/ACompArc/cvw
CVW_COMMON ?= $(CVW_ROOT)/examples/C/common

BLOCKS := 0 1 2 3 4 5 6 7
LOCAL_SOURCES := src/layernorm.c src/requantization.c src/projection.c src/attention.c src/residual.c src/gelu.c tests/ln2/test_allblocks_ln2.c \
    $(foreach b,$(BLOCKS),src/blocks/block$(b).c)
LOCAL_OBJECTS := $(addprefix $(BUILD_DIR)/,$(LOCAL_SOURCES:.c=.o))
PARAM_OBJECTS := $(foreach b,$(BLOCKS),$(BUILD_DIR)/params/block$(b)_params.o)
GOLDEN_OBJECTS := $(foreach b,$(BLOCKS),$(BUILD_DIR)/fixtures/block$(b)_ln2_golden.o) \
    $(BUILD_DIR)/fixtures/ln2_case_names.o
RUNTIME_OBJECTS := $(BUILD_DIR)/runtime/crt.o $(BUILD_DIR)/runtime/syscalls.o
OBJECTS := $(LOCAL_OBJECTS) $(PARAM_OBJECTS) $(GOLDEN_OBJECTS) $(RUNTIME_OBJECTS)
DEPS := $(OBJECTS:.o=.d)

CPPFLAGS += -I "$(SW_DIR)" -I "$(SW_DIR)/blocks" \
    -I "$(PARAMS_DIR)" -I "$(GOLDEN_DIR)"
CFLAGS := -std=gnu11 -O2 -Wall -Wextra \
    -march=$(ISA) -mabi=lp64 -mcmodel=medany \
    -fno-tree-loop-distribute-patterns
LDFLAGS := -march=$(ISA) -mabi=lp64 -mcmodel=medany \
    -nostdlib -static -T "$(CVW_COMMON)/test.ld"

ELF := $(BUILD_DIR)/test_allblocks_ln2.elf
DISASM := $(BUILD_DIR)/test_allblocks_ln2.dis

.PHONY: all riscv run inspect disasm clean
all: riscv
riscv: $(ELF)

$(BUILD_DIR)/src/%.o: $(SW_DIR)/%.c
	mkdir -p "$(@D)"
	$(RISCV_PREFIX)gcc $(CPPFLAGS) $(CFLAGS) -MMD -MP -c "$<" -o "$@"

$(BUILD_DIR)/tests/%.o: $(SW_TESTS_DIR)/%.c
	mkdir -p "$(@D)"
	$(RISCV_PREFIX)gcc $(CPPFLAGS) $(CFLAGS) -MMD -MP -c "$<" -o "$@"

$(BUILD_DIR)/params/%.o: $(PARAMS_DIR)/%.c
	mkdir -p "$(@D)"
	$(RISCV_PREFIX)gcc $(CPPFLAGS) $(CFLAGS) -MMD -MP -c "$<" -o "$@"

$(BUILD_DIR)/fixtures/%.o: $(GOLDEN_DIR)/%.c
	mkdir -p "$(@D)"
	$(RISCV_PREFIX)gcc $(CPPFLAGS) $(CFLAGS) -MMD -MP -c "$<" -o "$@"

$(BUILD_DIR)/runtime/crt.o: $(CVW_COMMON)/crt.S
	mkdir -p "$(@D)"
	$(RISCV_PREFIX)gcc $(CFLAGS) -I "$(CVW_COMMON)" -MMD -MP -c "$<" -o "$@"

$(BUILD_DIR)/runtime/syscalls.o: $(CVW_COMMON)/syscalls.c
	mkdir -p "$(@D)"
	$(RISCV_PREFIX)gcc $(CFLAGS) -I "$(CVW_COMMON)" -MMD -MP -c "$<" -o "$@"

$(ELF): $(OBJECTS) $(CVW_COMMON)/test.ld
	$(RISCV_PREFIX)gcc $(LDFLAGS) $(OBJECTS) -lgcc -o "$@"

run: $(ELF)
	$(SPIKE) --isa=$(ISA) "$<"

inspect: $(ELF)
	$(RISCV_PREFIX)readelf -A "$<"
	$(RISCV_PREFIX)size "$<"

disasm: $(DISASM)
$(DISASM): $(ELF)
	$(RISCV_PREFIX)objdump -d "$<" > "$@"

clean:
	rm -f $(OBJECTS) $(DEPS) "$(ELF)" "$(DISASM)"

-include $(DEPS)
