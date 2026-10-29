.DEFAULT_GOAL := riscv
.DELETE_ON_ERROR:
SHELL := /bin/bash
TEST_DIR := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
PROJECT_DIR := $(abspath $(TEST_DIR)/../../..)
SW_TESTS_DIR := $(PROJECT_DIR)/tests/sw
SW_DIR := $(PROJECT_DIR)/src/sw
BACKEND ?= custom
MODE ?= test
PROFILE_CASES ?= 1
ifneq ($(BACKEND),custom)
ifneq ($(BACKEND),scalar)
$(error BACKEND must be custom or scalar)
endif
endif
ifneq ($(MODE),test)
ifneq ($(MODE),profile)
$(error MODE must be test or profile)
endif
endif
PROFILE_FLAG := $(if $(filter profile,$(MODE)),1,0)
CUSTOM_FLAG := $(if $(filter custom,$(BACKEND)),1,0)
PARAMS_DIR ?= $(PROJECT_DIR)/build/parameter_export/c
GOLDEN_DIR ?= $(PROJECT_DIR)/build/fixtures/full
MODE_DIR := $(if $(filter profile,$(MODE)),profile$(PROFILE_CASES),test)
BUILD_DIR ?= $(PROJECT_DIR)/build/tests/full/$(BACKEND)/$(MODE_DIR)
RISCV_PREFIX ?= riscv64-unknown-elf-
SPIKE ?= spike
ISA ?= rv64imc_zicsr
CVW_ROOT ?= $(HOME)/Documents/ACompArc/cvw
CVW_COMMON ?= $(CVW_ROOT)/examples/C/common
BLOCKS := 0 1 2 3 4 5 6 7
BACKEND_SOURCES := $(if $(filter custom,$(BACKEND)),src/projection.c src/requantization.c,src/backends/scalar/projection.c src/backends/scalar/requantization.c)
LOCAL_SOURCES := src/model.c src/layernorm.c src/attention.c src/residual.c src/gelu.c \
    $(BACKEND_SOURCES) tests/full/test_allblocks.c $(foreach b,$(BLOCKS),src/blocks/block$(b).c)
LOCAL_OBJECTS := $(addprefix $(BUILD_DIR)/,$(LOCAL_SOURCES:.c=.o))
PARAM_OBJECTS := $(foreach b,$(BLOCKS),$(BUILD_DIR)/params/block$(b)_params.o)
OBJECTS := $(LOCAL_OBJECTS) $(PARAM_OBJECTS) $(BUILD_DIR)/fixtures/full_golden.o \
    $(BUILD_DIR)/runtime/crt.o $(BUILD_DIR)/runtime/syscalls.o
DEPS := $(OBJECTS:.o=.d)
CPPFLAGS += -I "$(SW_DIR)" -I "$(SW_DIR)/blocks" \
    -I "$(PARAMS_DIR)" -I "$(GOLDEN_DIR)" -DPROFILE_ONLY=$(PROFILE_FLAG) \
    -DPROFILE_CASE_COUNT=$(PROFILE_CASES) -DBUILD_CUSTOM_BACKEND=$(CUSTOM_FLAG)
CFLAGS := -std=gnu11 -O2 -Wall -Wextra -march=$(ISA) -mabi=lp64 -mcmodel=medany \
    -fno-tree-loop-distribute-patterns
LDFLAGS := -march=$(ISA) -mabi=lp64 -mcmodel=medany -nostdlib -static -T "$(CVW_COMMON)/test.ld"
ELF := $(BUILD_DIR)/test_allblocks.elf
DISASM := $(BUILD_DIR)/test_allblocks.dis
.PHONY: all riscv run disasm inspect clean
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
disasm: $(DISASM)
$(DISASM): $(ELF)
	$(RISCV_PREFIX)objdump -d "$<" > "$@"
inspect: $(ELF)
	$(RISCV_PREFIX)readelf -A "$<"
	$(RISCV_PREFIX)size "$<"
clean:
	rm -f $(OBJECTS) $(DEPS) "$(ELF)" "$(DISASM)"
-include $(DEPS)
