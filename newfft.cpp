#include <vector>
#include <iostream>
#include <string>
#include<complex>
#include <fstream>
#include <chrono>

using namespace std;
using namespace chrono;
typedef std::chrono::high_resolution_clock Clock;
#define PI 3.1415926535



void fft(vector<complex<double>>& input, int lim, int opt)
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
		complex<double> w_m(cos(PI / m), -1 * opt * sin(PI / m));

		for (int i = 0; i < n; i += k)
		{
			complex<double> w(1, 0);
			for (int j = 0; j < m; j++)
			{
				complex<double> t = w * input[i + j + m];
				input[i + j + m] = input[i + j] - t;
				input[i + j] = input[i + j] + t;
				w = w * w_m;
			}
		}
	}
	for (int i = 0; i < lim; i++)
	{
		if (opt == -1)
		{
			input[i] = input[i] / double(lim);
		}
		else
		{
			input[i] = input[i];
		}
	}
}


// 二维FFT函数
void FFT2D(vector<vector<complex<double>>>& src, vector<vector<complex<double>>>& dst, int opt)
{
    int height = src.size();
    int width = src[0].size();

    // 第一遍fft：对每一行应用一维FFT
    for (int i = 0; i < height; i++)
    {
        vector<complex<double>> tempData(width);
        // 获取每行数据
        for (int j = 0; j < width; j++)
        {
            tempData[j] = src[i][j];
        }
        // 一维FFT
        fft(tempData, width, opt);
        // 写入每行数据
        for (int j = 0; j < width; j++)
        {
            dst[i][j] = tempData[j];
        }
    }

    // 第二遍fft：对每一列应用一维FFT
    for (int i = 0; i < width; i++)
    {
        vector<complex<double>> tempData(height);
        // 获取每列数据
        for (int j = 0; j < height; j++)
        {
            tempData[j] = dst[j][i];
        }
        // 一维FFT
        fft(tempData, height, opt);
        // 写入每列数据
        for (int j = 0; j < height; j++)
        {
            dst[j][i] = tempData[j];
        }
    }
}

int main(int argc, char* argv[]) {
    std::ifstream fi("fft_8388608.txt");
    std::vector<double> data1;
    std::string read_temp;
    int count = 5;
    // Read input data from file
    while (fi.good()) {
        getline(fi, read_temp);
        data1.push_back(stod(read_temp));
    }
    fi.close();
    int N = data1.size();
    vector<complex<double>> fft_in(N);
    // Create input array dynamically

    for (size_t i = 0; i < N; i++)
    {
        fft_in[i] = complex<double>(data1[i], 0);
    }

    printf("Length of Sequence: %d\n", N);

    cout << "ready!" << std::endl;
    auto t1 = Clock::now();

    auto t2 = Clock::now(); // 计时结束
    cout << "fft cost : " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e+6 / count << " ms.\n";

    ofstream fo;
    //fo.open("fft_result.txt", ios::out);
    //for (int i = 0; i < data1.size(); i++)
    //{
    //    fo << '(' << fft_in[i].real() << ',' << fft_in[i].imag() << ')' << std::endl;
    //}
    //fo.close();
}



