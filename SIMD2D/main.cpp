#include <fstream>
#include <chrono>
#include <vector>
#include <iostream>
#include <string>
#include <cmath>
#include <immintrin.h>
#include <omp.h>
#include <complex>

#define N 8388608



const double PI = 3.14159265358979323846;

using namespace std;
using namespace chrono;
typedef std::chrono::high_resolution_clock Clock;

struct COMPLEX
{
    double real = 0;
    double imag = 0;
};

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

void fft_sse(double** data, int size, bool inverse = false)
{
    // 数据重排
     // Bit-reverse permutation
    double temp;
    int numBits = static_cast<int>(log2(size));
    for (int i = 1, j = 0; i < size; i++)
    {
        int bit = size >> 1;
        for (; j >= bit; bit >>= 1) j -= bit;
        j += bit;
        if (i < j)
        {
            temp = data[i][0];
            data[i][0] = data[j][0];
            data[j][0] = temp;

            temp = data[i][1];
            data[i][1] = data[j][1];
            data[j][1] = temp;
        }
    }
    // 蝴蝶运算
    for (int n = 2; n <= size; n <<= 1) {
        double wtemp = sin((PI / n) * (inverse ? -1 : 1));
        __m128d wr = _mm_set1_pd(1.0);
        __m128d wi = _mm_set1_pd(0.0);
        __m128d ir = _mm_set_pd(0, 1);
        __m128d wpi = _mm_set1_pd(sin((2 * PI / n) * (inverse ? -1 : 1)));
        __m128d wpr = _mm_set1_pd(-2.0 * wtemp * wtemp);
        for (int m = 0; m < n / 2; ++m) 
        {
            for (int i = m; i < size; i += n) 
            {
                __m128d angle = _mm_set_pd(-j * PI / m, j * PI / m);
                __m128d wr = _mm_cos_pd(angle);
                __m128d wi = _mm_sin_pd(angle);
                __m128d o = _mm_loadu_pd(data[i + j + m]);	// odd a|b
                __m128d e = _mm_loadu_pd(data[i + j]);	// even
                wr = _mm_mul_pd(o, wr);	// a*c|b*c
                __m128d n1 = _mm_shuffle_pd(o, o, _MM_SHUFFLE2(0, 1));
                wi = _mm_mul_pd(n1, wi);
                n1 = _mm_add_pd(wr, wi);
                wr = _mm_add_pd(e, n1);
                wi = _mm_sub_pd(e, n1);
                _mm_storeu_pd(data[i + j + m], wi);
                _mm_storeu_pd(data[i + j], wr);
            }
        }
    }
    // Scale if performing inverse FFT
    if (inverse) {
        for (int i = 0; i < size; ++i) {
            data[i][0] /= size;
            data[i][1] /= size;
        }
    }
}

void fft_omp(double** data, int size, bool inverse = false) {
    // 数据重排
     // Bit-reverse permutation
    double temp;
    int numBits = static_cast<int>(log2(size));
    for (int i = 1, j = 0; i < size; i++)
    {
        int bit = size >> 1;
        for (; j >= bit; bit >>= 1) j -= bit;
        j += bit;
        if (i < j)
        {
            temp = data[i][0];
            data[i][0] = data[j][0];
            data[j][0] = temp;

            temp = data[i][1];
            data[i][1] = data[j][1];
            data[j][1] = temp;
        }
    }
    // 蝴蝶运算
#pragma omp parallel num_threads(4)
    {
    for (int n = 2; n <= size; n <<= 1) {
        double angle = (2 * PI / n) * (inverse ? -1 : 1);
        double wtemp = sin(0.5 * angle);
        double wpr = -2.0 * wtemp * wtemp;
        double wpi = sin(angle);
        double wr = 1.0;
        double wi = 0.0;
#pragma omp for
        for (int m = 0; m < n / 2; ++m) {
            for (int i = m; i < size; i += n) {
                int j = i + n / 2;
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
#pragma omp for
        for (int i = 0; i < size; ++i) {
            data[i][0] /= size;
            data[i][1] /= size;
        }
    }
    }
}

void fft_avx512(double** data, int size, bool inverse = false)
{
    // 数据重排
     // Bit-reverse permutation
    double temp;
    int numBits = static_cast<int>(log2(size));
    for (int i = 1, j = 0; i < size; i++)
    {
        int bit = size >> 1;
        for (; j >= bit; bit >>= 1) j -= bit;
        j += bit;
        if (i < j)
        {
            temp = data[i][0];
            data[i][0] = data[j][0];
            data[j][0] = temp;

            temp = data[i][1];
            data[i][1] = data[j][1];
            data[j][1] = temp;
        }
    }
    // 蝴蝶运算
    {
        for (int k = 2; k <= 4; k <<= 1)
        {
            int m = k >> 1;
            for (int i = 0; i < size; i += k)
            {
                for (int j = 0; j < m; j++)
                {
                    // complex<double> t = w * input[i + j + m];
                    // compute t
                    __m128d angle = _mm_set_pd(-j * PI / m, j * PI / m);
                    //__m128d wr = _mm_cos_pd(angle);
                    //__m128d wi = _mm_sin_pd(angle);
                    __m128d wr = _mm_set_pd(0, 0);
                    __m128d wi = _mm_set_pd(0, 0);
                    __m128d o = _mm_loadu_pd(data[i + j + m]);	// odd a|b
                    __m128d e = _mm_loadu_pd(data[i + j]);	// even
                    wr = _mm_mul_pd(o, wr);	// a*c|b*c
                    __m128d n1 = _mm_shuffle_pd(o, o, _MM_SHUFFLE2(0, 1));
                    wi = _mm_mul_pd(n1, wi);
                    n1 = _mm_add_pd(wr, wi);
                    wr = _mm_add_pd(e, n1);
                    wi = _mm_sub_pd(e, n1);
                    // input[i + j + m] = input[i + j] - t;
                    _mm_storeu_pd(data[i + j + m], wi);
                    // input[i + j] += t;
                    _mm_storeu_pd(data[i + j], wr);
                }
            }
        }

        for (int k = 8; k <= size; k <<= 1) // 
        {
            int m = k >> 1;
            for (int i = 0; i < size; i += k)
            {
                for (int j = 0; j < m; j += 4) // 
                {
                    // complex<double> t = w * input[i + j + m];
                    // compute t
                    __m512d angle = _mm512_set_pd(
                        -PI * (j + 3) / m, PI * (j + 3) / m,
                        -PI * (j + 2) / m, PI * (j + 2) / m,
                        -PI * (j + 1) / m, PI * (j + 1) / m,
                        -PI * j / m, PI * j / m
                    );

                    __m512d wr = _mm512_cos_pd(angle);
                    __m512d wi = _mm512_sin_pd(angle);
                    __m512d o = _mm512_loadu_pd(data[i + j + m]);	// odd a|b
                    __m512d e = _mm512_loadu_pd(data[i + j]);	// even
                    wr = _mm512_mul_pd(o, wr);	// a*c|b*c
                    // __m512d n1 = _mm512_shuffle_pd(o, o, 0x55);
                    __m512d n1 = _mm512_shuffle_pd(o, o, 0x55);
                    wi = _mm512_mul_pd(n1, wi);
                    n1 = _mm512_add_pd(wr, wi);
                    wr = _mm512_add_pd(e, n1);
                    wi = _mm512_sub_pd(e, n1);
                    // input[i + j + m] = input[i + j] - t;
                    _mm512_storeu_pd(data[i + j + m], wi);
                    // input[i + j] += t;
                    _mm512_storeu_pd(data[i + j], wr);
                }
            }
        }
    }
    // Scale if performing inverse FFT
    if (inverse) {
        for (int i = 0; i < size; ++i) {
            data[i][0] /= size;
            data[i][1] /= size;
        }
    }
}

void fft2_sse(COMPLEX* input, int length)
{
    // 数据重排
    for (int i = 1, j = 0; i < length; i++)
    {
        int bit = length >> 1;
        for (; j >= bit; bit >>= 1) j -= bit;
        j += bit;
        if (i < j) swap(input[i], input[j]);
    }
    // 蝴蝶运算
    for (int k = 2; k <= length; k <<= 1)
    {
        int m = k >> 1;
        for (int i = 0; i < length; i += k)
        {
            for (int j = 0; j < m; j++)
            {
                // complex<double> t = w * input[i + j + m];
                // compute t
                __m128d angle = _mm_set_pd(-j * PI / m, j * PI / m);
                __m128d wr = _mm_cos_pd(angle);
                __m128d wi = _mm_sin_pd(angle);
                //__m128d wr = _mm_set1_pd(cos(j * PI / m));
                //__m128d wi = _mm_set_pd(-sin(j * PI / m), sin(j * PI / m));
                __m128d o = _mm_load_pd((double*)&input[i + j + m]);	// odd a|b
                __m128d e = _mm_load_pd((double*)&input[i + j]);	// even
                wr = _mm_mul_pd(o, wr);	// a*c|b*c
                __m128d n1 = _mm_shuffle_pd(o, o, _MM_SHUFFLE2(0, 1));
                wi = _mm_mul_pd(n1, wi);
                n1 = _mm_add_pd(wr, wi);
                wr = _mm_add_pd(e, n1);
                wi = _mm_sub_pd(e, n1);
                // input[i + j + m] = input[i + j] - t;
                _mm_store_pd((double*)&input[i + j + m], wi);
                // input[i + j] += t;
                _mm_store_pd((double*)&input[i + j], wr);
            }
        }
    }
}

void fft2(vector<complex<double>>& input)
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
        complex<double> w_m(cos(PI / m), -sin(PI / m));

        for (int i = 0; i < n; i += k)
        {
            complex<double> w(1);
            for (int j = 0; j < m; j++)
            {
                complex<double> t = w * input[i + j + m];
                input[i + j + m] = input[i + j] - t;
                input[i + j] += t;
                w *= w_m;
            }
        }
    }

}

void fft2_avx512(COMPLEX* input, int length)
{
    // 数据重排
    for (int i = 1, j = 0; i < length; i++)
    {
        int bit = length >> 1;
        for (; j >= bit; bit >>= 1) j -= bit;
        j += bit;
        if (i < j) swap(input[i], input[j]);
    }
    // 蝴蝶运算
    {
        for (int k = 2; k <= 4; k <<= 1)
        {
            int m = k >> 1;
            for (int i = 0; i < length; i += k)
            {
                for (int j = 0; j < m; j++)
                {
                    // complex<double> t = w * input[i + j + m];
                    // compute t
                    __m128d wr = _mm_set1_pd(cos(j * PI / m));
                    __m128d wi = _mm_set_pd(-sin(j * PI / m), sin(j * PI / m));
                    __m128d o = _mm_load_pd((double*)&input[i + j + m]);	// odd a|b
                    __m128d e = _mm_load_pd((double*)&input[i + j]);	// even
                    wr = _mm_mul_pd(o, wr);	// a*c|b*c
                    __m128d n1 = _mm_shuffle_pd(o, o, _MM_SHUFFLE2(0, 1));
                    wi = _mm_mul_pd(n1, wi);
                    n1 = _mm_add_pd(wr, wi);
                    wr = _mm_add_pd(e, n1);
                    wi = _mm_sub_pd(e, n1);
                    // input[i + j + m] = input[i + j] - t;
                    _mm_store_pd((double*)&input[i + j + m], wi);
                    // input[i + j] += t;
                    _mm_store_pd((double*)&input[i + j], wr);
                }
            }
        }

        for (int k = 8; k <= length; k <<= 1) // 
        {
            int m = k >> 1;
            for (int i = 0; i < length; i += k)
            {
                for (int j = 0; j < m; j+=4) // 
                {
                    // complex<double> t = w * input[i + j + m];
                    // compute t
                    __m512d angle = _mm512_set_pd(
                        -PI * (j + 3) / m, PI * (j + 3) / m,
                        -PI * (j + 2) / m, PI * (j + 2) / m,
                        -PI * (j + 1) / m, PI * (j + 1) / m,
                        -PI * j / m, PI * j / m
                    );

                    __m512d wr = _mm512_cos_pd(angle);
                    __m512d wi = _mm512_sin_pd(angle);
                    __m512d o = _mm512_load_pd((double*)&input[i + j + m]);	// odd a|b
                    __m512d e = _mm512_load_pd((double*)&input[i + j]);	// even
                    wr = _mm512_mul_pd(o, wr);	// a*c|b*c
                    // __m512d n1 = _mm512_shuffle_pd(o, o, 0x55);
                    __m512d n1 = _mm512_shuffle_pd(o, o, 0x55);
                    wi = _mm512_mul_pd(n1, wi);
                    n1 = _mm512_add_pd(wr, wi);
                    wr = _mm512_add_pd(e, n1);
                    wi = _mm512_sub_pd(e, n1);
                    // input[i + j + m] = input[i + j] - t;
                    _mm512_store_pd((double*)&input[i + j + m], wi);
                    // input[i + j] += t;
                    _mm512_store_pd((double*)&input[i + j], wr);
                }
            }
        }
    }
}

void fft2_avx512_openmp(COMPLEX* input, int length)
{
    // 数据重排
    for (int i = 1, j = 0; i < length; i++)
    {
        int bit = length >> 1;
        for (; j >= bit; bit >>= 1) j -= bit;
        j += bit;
        if (i < j) swap(input[i], input[j]);
    }
    // 蝴蝶运算
#pragma omp parallel num_threads(4)
    {
        for (int k = 2; k <= 4; k <<= 1)
        {
            int m = k >> 1;
#pragma omp for 
            for (int i = 0; i < length; i += k)
            {
                for (int j = 0; j < m; j++)
                {
                    // complex<double> t = w * input[i + j + m];
                    // compute t
                    __m128d wr = _mm_set1_pd(cos(j * PI / m));
                    __m128d wi = _mm_set_pd(-sin(j * PI / m), sin(j * PI / m));
                    __m128d o = _mm_load_pd((double*)&input[i + j + m]);	// odd a|b
                    __m128d e = _mm_load_pd((double*)&input[i + j]);	// even
                    wr = _mm_mul_pd(o, wr);	// a*c|b*c
                    __m128d n1 = _mm_shuffle_pd(o, o, _MM_SHUFFLE2(0, 1));
                    wi = _mm_mul_pd(n1, wi);
                    n1 = _mm_add_pd(wr, wi);
                    wr = _mm_add_pd(e, n1);
                    wi = _mm_sub_pd(e, n1);
                    // input[i + j + m] = input[i + j] - t;
                    _mm_store_pd((double*)&input[i + j + m], wi);
                    // input[i + j] += t;
                    _mm_store_pd((double*)&input[i + j], wr);
                }
            }
        }

        for (int k = 8; k <= length; k <<= 1) // 
        {
            int m = k >> 1;
#pragma omp for 
            for (int i = 0; i < length; i += k)
            {
                for (int j = 0; j < m; j += 4) // 
                {
                    // complex<double> t = w * input[i + j + m];
                    // compute t
                    __m512d angle = _mm512_set_pd(
                        -PI * (j + 3) / m, PI * (j + 3) / m,
                        -PI * (j + 2) / m, PI * (j + 2) / m,
                        -PI * (j + 1) / m, PI * (j + 1) / m,
                        -PI * j / m, PI * j / m
                    );

                    __m512d wr = _mm512_cos_pd(angle);
                    __m512d wi = _mm512_sin_pd(angle);
                    __m512d o = _mm512_load_pd((double*)&input[i + j + m]);	// odd a|b
                    __m512d e = _mm512_load_pd((double*)&input[i + j]);	// even
                    wr = _mm512_mul_pd(o, wr);	// a*c|b*c
                    // __m512d n1 = _mm512_shuffle_pd(o, o, 0x55);
                    __m512d n1 = _mm512_shuffle_pd(o, o, 0x55);
                    wi = _mm512_mul_pd(n1, wi);
                    n1 = _mm512_add_pd(wr, wi);
                    wr = _mm512_add_pd(e, n1);
                    wi = _mm512_sub_pd(e, n1);
                    // input[i + j + m] = input[i + j] - t;
                    _mm512_store_pd((double*)&input[i + j + m], wi);
                    // input[i + j] += t;
                    _mm512_store_pd((double*)&input[i + j], wr);
                }
            }
        }
    }
}



int main() {
    std::ifstream fi("fft_8388608.txt");
    std::vector<double> data1;
    std::string read_temp;
    int count = 10;
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
    COMPLEX *data2 = new COMPLEX[N];
    for (int i = 0; i < N; i++)
    {
        data2[i].real = data1[i];
        data2[i].imag = 0.0;

    }
    printf("Length of Sequence: %d\n", N);

    // warm up
    for (int i = 0; i < 1; i++)
    {
        fft_sse(data, N);
    }

    // Perform forward FFT
    auto t1 = Clock::now();
    for (int i = 0; i < count; i++)
    {
        fft_sse(data, N);
    }
    auto t2 = Clock::now(); // 计时结束
    cout << "fft cost : " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e+6 / count << " ms.\n";

    ofstream fo;
    fo.open("fft_result.txt", ios::out);
    for (int i = 0; i < data1.size(); i++)
    {
        fo << '(' << data[i][0] << ',' << data[i][1] << ')' << endl;
    }
    fo.close();
    //ofstream fo;
    //fo.open("fft_result.txt", ios::out);
    //for (int i = 0; i < data1.size(); i++)
    //{
    //    fo << '(' << data2[i].real << ',' << data2[i].imag << ')' << endl;
    //}
    //fo.close();
    // Clean up
    for (int i = 0; i < N; i++) {
        delete[] data[i];
    }
    delete[] data;
    return 0;
}