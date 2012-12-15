EXECUTABLES = print-devices solve solve_float

all: $(EXECUTABLES)

ifdef OPENCL_INC
  CL_CFLAGS = -I$(OPENCL_INC)
endif

ifdef OPENCL_LIB
  CL_LDFLAGS = -L$(OPENCL_LIB)
endif

solve: solve.c cl-helper.o 
	gcc $(CL_CFLAGS) $(CL_LDFLAGS) -std=gnu99 -lrt -lOpenCL -o$@ $^

solve_float: solve_float.c cl-helper.o 
	gcc $(CL_CFLAGS) $(CL_LDFLAGS) -std=gnu99 -lrt -lOpenCL -o$@ $^

print-devices: print-devices.c cl-helper.o
	gcc $(CL_CFLAGS) $(CL_LDFLAGS) -std=gnu99 -lrt -lOpenCL -o$@ $^

%.o : %.c %.h
	gcc -c $(CL_CFLAGS) -std=gnu99 $<

clean:
	rm -f $(EXECUTABLES) *.o
