//==============================================================
// Vector Add is the equivalent of a Hello, World! sample for data parallel
// programs. Building and running the sample verifies that your development
// environment is setup correctly and demonstrates the use of the core features
// of SYCL. This sample runs on both CPU and GPU (or FPGA). When run, it
// computes on both the CPU and offload device, then compares results. If the
// code executes on both CPU and offload device, the device name and a success
// message are displayed. And, your development environment is setup correctly!
//
// For comprehensive instructions regarding SYCL Programming, go to
// https://software.intel.com/en-us/oneapi-programming-guide and search based on
// relevant terms noted in the comments.
//
// SYCL material used in the code sample:
// ?A one dimensional array of data.
// ?A device queue, buffer, accessor, and kernel.
//==============================================================
// Copyright ?Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <sycl/sycl.hpp>
#include <vector>
#include <iostream>
#include <string>
#include<complex>
#include <chrono>
#if FPGA || FPGA_EMULATOR
#include <sycl/ext/intel/fpga_extensions.hpp>
#endif
using namespace std;
using namespace sycl;
using namespace chrono;


typedef std::chrono::high_resolution_clock Clock;

#define PI 3.1415926535
// num_repetitions: How many times to repeat the kernel invocation
size_t num_repetitions = 1;
// Vector type and data size for this example.
size_t vector_size = 10000;
typedef std::vector<int> IntVector;

// Create an exception handler for asynchronous SYCL exceptions
static auto exception_handler = [](sycl::exception_list e_list) {
    for (std::exception_ptr const& e : e_list) {
        try {
            std::rethrow_exception(e);
        }
        catch (std::exception const& e) {
#if _DEBUG
            std::cout << "Failure" << std::endl;
#endif
            std::terminate();
        }
    }
};

//************************************
// Vector add in SYCL on device: returns sum in 4th parameter "sum_parallel".
//************************************
void VectorAdd(queue& q, const IntVector& a_vector, const IntVector& b_vector,
    IntVector& sum_parallel) {
    // Create the range object for the vectors managed by the buffer.
    range<1> num_items{ a_vector.size() };

    // Create buffers that hold the data shared between the host and the devices.
    // The buffer destructor is responsible to copy the data back to host when it
    // goes out of scope.
    buffer a_buf(a_vector);
    buffer b_buf(b_vector);
    buffer sum_buf(sum_parallel.data(), num_items);

    for (size_t i = 0; i < num_repetitions; i++) {

        // Submit a command group to the queue by a lambda function that contains the
        // data access permission and device computation (kernel).
        q.submit([&](handler& h) {
            // Create an accessor for each buffer with access permission: read, write or
            // read/write. The accessor is a mean to access the memory in the buffer.
            accessor a(a_buf, h, read_only);
            accessor b(b_buf, h, read_only);

            // The sum_accessor is used to store (with write permission) the sum data.
            accessor sum(sum_buf, h, write_only, no_init);

            // Use parallel_for to run vector addition in parallel on device. This
            // executes the kernel.
            //    1st parameter is the number of work items.
            //    2nd parameter is the kernel, a lambda that specifies what to do per
            //    work item. The parameter of the lambda is the work item id.
            // SYCL supports unnamed lambda kernel by default.
            h.parallel_for(num_items, [=](auto i) { sum[i] = a[i] + b[i]; });
            });
    };
    // Wait until compute tasks on GPU done
    q.wait();
}


void InitializeVector(IntVector& a) 
{
    for (size_t i = 0; i < a.size(); i++) a.at(i) = i;
}

void fft(vector<complex<float>>& input)//迭代算法 
{
	int n = input.size();

	// 数据重排
	for (int i = 1, j = 0; i < n; i++) 
	{
		int bit = n >> 1;
		for (; j >= bit; bit >>= 1) j -= bit;
		j += bit;
		if (i < j) swap(input[i], input[j]);
	}
	// 蝴蝶运算
	for (int k = 2; k <= n; k <<= 1) 
	{
		int m = k >> 1;
		complex<float> w_m(cos(PI / m), -sin(PI / m));

		for (int i = 0; i < n; i += k) 
		{
			complex<float> w(1);
			for (int j = 0; j < m; j++) 
			{
				complex<float> t = w * input[i + j + m];
				input[i + j + m] = input[i + j] - t;
				input[i + j] += t;
				w *= w_m;
			}
		}
	}
}


void fft2(vector<complex<float>>& input, sycl::queue& q) {
    int n = input.size();
    
    // 数据重排
    for (int i = 1, j = 0; i < n; i++)
    {
        int bit = n >> 1;
        for (; j >= bit; bit >>= 1) j -= bit;
        j += bit;
        if (i < j) swap(input[i], input[j]);
    }
    
    
    // Create a SYCL buffer to hold the input data
    buffer<complex<float>, 1> input_buffer(input.data(), range<1>(n));

    // Data reordering (parallelize this part)
  
        // Data reordering (parallelize this part)

    // Butterfly operation (parallelize this part)
    
    for (int k = 2; k <= n; k <<= 1) {
        int m = k >> 1;
        complex<float> w_m(cos(PI / m), -sin(PI / m));



        q.submit([&](handler& h) {
            auto in = input_buffer.get_access<access::mode::read_write>(h);
            h.parallel_for(range<1>(n), [=](id<1> idx)
                {
                int i = idx.get(0);
                int group_idx = i / k;//索引的组号
                int j = i % k;//组内索引
                //complex<float> w(1);
                complex<float> w = pow(w_m, j);
               
                if (j<m) {
                    complex<float> t = w * in[group_idx * k + j+m];
                    in[group_idx * k + j+m] = in[group_idx * k + j ] - t;
                    in[group_idx * k + j] += t;
                }
               // w *= w_m;
                });
            });


  q.wait();

    }
      
}


int main(int argc, char* argv[]) {
   std::ifstream fi("oneAPI_Essentials/02_SYCL_Program_Structure/fft_8388608.txt");
    std::vector<float> data1;
    std::string read_temp;
    int count = 100;
    // Read input data from file
    while (fi.good()) {
        getline(fi, read_temp);
        data1.push_back(stod(read_temp));
    }
    fi.close();
    int N = data1.size();
    vector<complex<float>> fft_in(N);



    for (size_t i = 0; i < N; i++)
    {
        fft_in[i] = complex<float>(data1[i], 0);
    }
    //default_selector selector;
    //queue q(selector);
   // queue q{ gpu_selector() };
   // auto propList = sycl::property_list{sycl::property::queue::enable_profiling() };
    queue my_gpu_queue(sycl::cpu_selector_v);

    
    for (int i = 0; i < 5; i++)
    {
        fft(fft_in);
    }
    auto t1 = Clock::now();

    for (int i = 0; i < count; i++)
    {
        fft(fft_in);
    }

    auto t2 = Clock::now();
    cout << "fft_oneapi cost : " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e+6 / count << " ms.\n";

    ofstream fo;
    fo.open("fft_result.txt", ios::out);
    for (int i = 0; i < data1.size(); i++)
    {
        fo << '(' << fft_in[i].real() << ',' << fft_in[i].imag() << ')' << std::endl;
    }
    fo.close();
    cout<<"hello world!"<<std::endl;

}
