#pragma comment(lib, "libfftw3-3.lib")
#pragma comment(lib, "libfftw3f-3.lib")
#pragma comment(lib, "libfftw3l-3.lib")

#include "fftw3.h"
#include <fstream>
#include <chrono>
#include <vector>
#include <iostream>
#include <string>
#include <cmath>

#define N 4194304

const double PI = 3.14159265358979323846;

using namespace std;
using namespace chrono;
typedef std::chrono::high_resolution_clock Clock;

// Function to perform FFT
void fft(double** data, int size, bool inverse = false) {
    // Bit-reverse permutation
    int i, j, k, n, m;
    double temp, angle;
    int numBits = static_cast<int>(log2(size));

    for (i = 0; i < size; ++i) {
        j = 0;
        for (k = 0; k < numBits; ++k)
            j = (j << 1) | ((i >> k) & 1);

        if (i < j) {
            temp = data[i][0];
            data[i][0] = data[j][0];
            data[j][0] = temp;

            temp = data[i][1];
            data[i][1] = data[j][1];
            data[j][1] = temp;
        }
    }

    // Cooley-Tukey algorithm
    for (n = 2; n <= size; n <<= 1) {
        angle = (2 * PI / n) * (inverse ? -1 : 1);
        double wtemp = sin(0.5 * angle);
        double wpr = -2.0 * wtemp * wtemp;
        double wpi = sin(angle);
        double wr = 1.0;
        double wi = 0.0;

        for (m = 0; m < n / 2; ++m) {
            for (i = m; i < size; i += n) {
                j = i + n / 2;
                double tempr = wr * data[j][0] - wi * data[j][1];
                double tempi = wr * data[j][1] + wi * data[j][0];
                data[j][0] = data[i][0] - tempr;
                data[j][1] = data[i][1] - tempi;
                data[i][0] += tempr;
                data[i][1] += tempi;
            }
            temp = wr;
            wr += wr * wpr - wi * wpi;
            wi += wi * wpr + temp * wpi;
        }
    }

    // Scale if performing inverse FFT
    if (inverse) {
        for (i = 0; i < size; ++i) {
            data[i][0] /= size;
            data[i][1] /= size;
        }
    }
}


//int main()
//{
//    fftw_complex* din, * out;
//    fftw_plan p;
//    din = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);
//    out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);
//    if ((din == NULL) || (out == NULL))
//    {
//        printf("Error:insufficient available memory\n");
//        return -1;
//    }
//    int count = 100;
//    ifstream fi("fft_8388608.txt");
//    vector<double> data;
//    string read_temp;
//    while (fi.good()) {
//        getline(fi, read_temp);
//        data.push_back(stod(read_temp));
//    }
//    fi.close();
//
//    for (int i = 0; i < N; ++i) {
//            din[i][0] = data[i];
//            din[i][1] = 0;
//    }
//    printf("Length of Sequence: %d\n", N);
//    
//    // warm up
//    p = fftw_plan_dft_1d(N, din, out, FFTW_FORWARD, FFTW_ESTIMATE);
//    fftw_execute(p); /* repeat as needed */
//    fftw_destroy_plan(p);
//    fftw_cleanup();
//    
//
//    auto t1 = Clock::now();
//    for (int i = 0; i < count; i++) {
//        // auto t3 = Clock::now(); // 计时结束
//        p = fftw_plan_dft_1d(N, din, out, FFTW_FORWARD, FFTW_ESTIMATE);
//        fftw_execute(p); /* repeat as needed */
//        fftw_destroy_plan(p);
//        fftw_cleanup();
//        auto t4 = Clock::now(); // 计时结束
//        // cout << "fftw cost : " << std::chrono::duration_cast<std::chrono::nanoseconds>(t4 - t3).count() / 1e+6 << " ms.\n";
//    }
//    auto t2 = Clock::now(); // 计时结束
//    cout << "fftw cost : " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e+6 /count << " ms.\n";
//    //ofstream fo;
//    //fo.open("fftw_result.txt", ios::out);
//    //for (int i = 0; i < data.size(); i++)
//    //{
//    //    fo << '(' << out[i][0] << ',' << out[i][1] << ')' << endl;
//    //}
//    //fo.close();
//    if (din != NULL) fftw_free(din);
//    if (out != NULL) fftw_free(out);
//    return 0;
//}

int main() {
    std::ifstream fi("fft_4194304.txt");
    std::vector<double> data1;
    std::string read_temp;
    int count = 1;
    // Read input data from file
    while (fi.good()) {
        getline(fi, read_temp);
        data1.push_back(stod(read_temp));
    }
    fi.close();

    // Create input array dynamically
    double** data = new double* [N];
    for (int i = 0; i < N; i++) {
        data[i] = new double[2];
        data[i][0] = data1[i];  // Real part
        data[i][1] = 0.0;       // Imaginary part
    }
    printf("Length of Sequence: %d\n", N);
    // Perform forward FFT
    auto t1 = Clock::now();
    for (int i = 0; i < count; i++)
    {
        fft(data, N);
    }
    auto t2 = Clock::now(); // 计时结束
    cout << "fft cost : " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e+6 /count << " ms.\n";

    //ofstream fo;
    //fo.open("fftw_result.txt", ios::out);
    //for (int i = 0; i < data1.size(); i++)
    //{
    //    fo << '(' << data[i][0] << ',' << data[i][1] << ')' << endl;
    //}
    //fo.close();
// 
    // Clean up
    for (int i = 0; i < N; i++) {
        delete[] data[i];
    }
    delete[] data;

    return 0;
}