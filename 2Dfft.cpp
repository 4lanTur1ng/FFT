#include <iostream>
#include<opencv2/opencv.hpp>
#include<omp.h>
using namespace cv;
using namespace std;
using namespace chrono;
typedef std::chrono::high_resolution_clock Clock;

const int height = 512, width = 512;

const double PI = acos(-1); // piֵ


struct Cpx // ����һ�������ṹ��͸������㷨��
{
	double r, i;
	Cpx() : r(0), i(0) {}
	Cpx(double _r, double _i) : r(_r), i(_i) {}
};
Cpx operator + (Cpx a, Cpx b) { return Cpx(a.r + b.r, a.i + b.i); }
Cpx operator - (Cpx a, Cpx b) { return Cpx(a.r - b.r, a.i - b.i); }
Cpx operator * (Cpx a, Cpx b) { return Cpx(a.r * b.r - a.i * b.i, a.r * b.i + a.i * b.r); }
Cpx operator / (Cpx a, int b) { return Cpx(a.r * 1.0 / b, a.i * 1.0 / b); }

Mat BGR2GRAY(Mat img)
{
	int w = img.cols;
	int h = img.rows;
	Mat grayImg(h, w, CV_8UC1);

	uchar* p = grayImg.ptr<uchar>(0);
	Vec3b* pImg = img.ptr<Vec3b>(0);

	for (int i = 0; i < w * h; ++i)
	{
		p[i] = 0.2126 * pImg[i][2] + 0.7152 * pImg[i][1] + 0.0722 * pImg[i][0];
	}
	return grayImg;
}

Mat Resize(Mat img)
{
	int w = img.cols;
	int h = img.rows;
	Mat out(height, width, CV_8UC1);
	uchar* p = out.ptr<uchar>(0);
	uchar* p2 = img.ptr<uchar>(0);
	int x_before, y_before;
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++)
		{
			x_before = (int)x * w * 1.0 / width;
			y_before = (int)y * h * 1.0 / height;
			p[y * width + x] = p2[y_before * w + x_before];
		}
	}
	return out;

}

//�Զ����ŵ�0-255��Χ��,���任���ޣ�����Ƶ�����м�
int AutoScale(Mat src, Mat out)
{
	int w = src.cols;
	int h = src.rows;
	double* p = src.ptr<double>(0);
	uchar* pOut = out.ptr<uchar>(0);
	double max = p[0];
	double min = p[0];
	for (int i = 0; i < w * h; i++)
	{
		if (p[i] > max) max = p[i];
		if (p[i] < min) min = p[i];
	}

	double scale = 255.0 / (max - min);

	for (int i = 0; i < w * h; i++)
	{
		int j = i + w * h / 2 + w / 2;
		if (j > w * h) j = j - w * h;   //��Ƶ�����м�
		pOut[i] = (uchar)((p[j] - min) * scale);
	}
	return 0;
}

void fft(vector<Cpx>& a, int lim, int opt)
{
	if (lim == 1) return;
	vector<Cpx> a0(lim >> 1), a1(lim >> 1); // ��ʼ��һ���С�����ż������������
	for (int i = 0; i < lim; i += 2)
		a0[i >> 1] = a[i], a1[i >> 1] = a[i + 1]; // �ֳ�ż�����ֺ���������

	fft(a0, lim >> 1, opt); // �ݹ����ż������
	fft(a1, lim >> 1, opt); // �ݹ����ż������

	Cpx wn(cos(2 * PI / lim), opt * -sin(2 * PI / lim)); //����WN
	Cpx w(1, 0);
	for (int k = 0; k < (lim >> 1); k++) // ������ͼ1�������
	{
		a[k] = a0[k] + w * a1[k];
		a[k + (lim >> 1)] = a0[k] - w * a1[k];
		w = w * wn;
	}

	//for (int k = 0; k < (lim >> 1); k++) // ������ͼ2��С�Ż�һ�£���һ�γ˷�
	//{
	//	Cpx t = w * a1[k];
	//	a[k] = a0[k] + t;
	//	a[k + (lim >> 1)] = a0[k] - t;
	//	w = w * wn;
	//}

}

//��������������
int ReverseBin(int a, int n)
{
	int ret = 0;
	for (int i = 0; i < n; i++)
	{
		if (a & (1 << i)) ret |= (1 << (n - 1 - i));
	}
	return ret;
}

void fft2(vector<Cpx>& a, int lim, int opt)
{
	int index;
	vector<Cpx> tempA(lim);
	for (int i = 0; i < lim; i++)
	{
		index = ReverseBin(i, log2(lim));
		tempA[i] = a[index];
	}

	vector<Cpx> WN(lim / 2);
	//����WN��,�����ظ�����
	for (int i = 0; i < lim / 2; i++)
	{
		WN[i] = Cpx(cos(2 * PI * i / lim), opt * -sin(2 * PI * i / lim));
	}

	//��������
	int Index0, Index1;
	Cpx temp;
	for (int steplenght = 2; steplenght <= lim; steplenght *= 2)
	{
		for (int step = 0; step < lim / steplenght; step++)
		{
			for (int i = 0; i < steplenght / 2; i++)
			{
				Index0 = steplenght * step + i;
				Index1 = steplenght * step + i + steplenght / 2;

				temp = tempA[Index1] * WN[lim / steplenght * i];
				tempA[Index1] = tempA[Index0] - temp;
				tempA[Index0] = tempA[Index0] + temp;
			}
		}
	}
	for (int i = 0; i < lim; i++)
	{
		if (opt == -1)
		{
			a[i] = tempA[i] / lim;
		}
		else
		{
			a[i] = tempA[i];
		}
	}
}


void fft_recur(vector<Cpx>& input, int lim, int opt)//�����㷨 
{
	int n = input.size();

	// ��������
	for (int i = 1, j = 0; i < n; i++)
	{
		int bit = n >> 1;
		for (; j >= bit; bit >>= 1) j -= bit;
		j += bit;
		if (i < j) swap(input[i], input[j]);
	}
	// ��������
	for (int k = 2; k <= n; k <<= 1)
	{
		int m = k >> 1;
		Cpx w_m(cos(PI / m), -1*opt*sin(PI / m));

		for (int i = 0; i < n; i += k)
		{
			Cpx w(1,0);
			for (int j = 0; j < m; j++)
			{
				Cpx t = w * input[i + j + m];
				input[i + j + m] = input[i + j] - t;
				input[i + j] = input[i + j]+ t;
				w = w *w_m;
			}
		}
	}

	
	for (int i = 0; i < lim; i++)
	{
		if (opt == -1)
		{
			input[i] = input[i] / lim;
		}
		else
		{
			input[i] = input[i];
		}
	}
	
}

void fft_recur_simd(vector<Cpx>& input, int length, int opt)//�����㷨 
{
	// ��������
	for (int i = 1, j = 0; i < length; i++)
	{
		int bit = length >> 1;
		for (; j >= bit; bit >>= 1) j -= bit;
		j += bit;
		if (i < j) swap(input[i], input[j]);
	}
	// ��������
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

void fft_recur_simd_omp(vector<Cpx>& input, int length, int opt)//�����㷨 
{
	// ��������
	for (int i = 1, j = 0; i < length; i++)
	{
		int bit = length >> 1;
		for (; j >= bit; bit >>= 1) j -= bit;
		j += bit;
		if (i < j) swap(input[i], input[j]);
	}
	// ��������
	for (int k = 2; k <= length; k <<= 1)
	{
		int m = k >> 1;
#pragma omp parallel for num_threads(2)
		for (int i = 0; i < length; i += k)
		{
			for (int j = 0; j < m; j++)
			{
				// complex<double> t = w * input[i + j + m];
				// compute t
				__m128d angle = _mm_set_pd(-j * PI / m, j * PI / m);
				__m128d wr = _mm_cos_pd(angle);
				__m128d wi = _mm_sin_pd(angle);
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
void FFT2D_simd_omp(Cpx(*src)[width], Cpx(*dst)[width], int opt)
{
	//��һ��fft
#pragma omp parallel num_threads(4)
	{
#pragma omp for
		for (int i = 0; i < height; i++)
		{
			vector<Cpx> tempData(width);
			//��ȡÿ������
			for (int j = 0; j < width; j++)
			{
				tempData[j] = src[i][j];
			}
			//һάFFT
			fft_recur_simd_omp(tempData, width, opt);
			//д��ÿ������
			for (int j = 0; j < width; j++)
			{
				dst[i][j] = tempData[j];
			}
		}

		//�ڶ���fft
#pragma omp for
		for (int i = 0; i < width; i++)
		{
			vector<Cpx> tempData(height);
			//��ȡÿ������
			for (int j = 0; j < height; j++)
			{
				tempData[j] = dst[j][i];
			}
			//һάFFT
			fft_recur_simd_omp(tempData, height, opt);
			//д��ÿ������
			for (int j = 0; j < height; j++)
			{
				dst[j][i] = tempData[j];
			}
		}


	}
	
}





void FFT2D_simd(Cpx(*src)[width], Cpx(*dst)[width], int opt)
{
	//��һ��fft
	for (int i = 0; i < height; i++)
	{
		vector<Cpx> tempData(width);
		//��ȡÿ������
		for (int j = 0; j < width; j++)
		{
			tempData[j] = src[i][j];
		}
		//һάFFT
		fft_recur_simd(tempData, width, opt);
		//д��ÿ������
		for (int j = 0; j < width; j++)
		{
			dst[i][j] = tempData[j];
		}
	}

	//�ڶ���fft
	for (int i = 0; i < width; i++)
	{
		vector<Cpx> tempData(height);
		//��ȡÿ������
		for (int j = 0; j < height; j++)
		{
			tempData[j] = dst[j][i];
		}
		//һάFFT
		fft_recur_simd(tempData, height, opt);
		//д��ÿ������
		for (int j = 0; j < height; j++)
		{
			dst[j][i] = tempData[j];
		}
	}
}

void Mat2Cpx(Mat src, Cpx(*dst)[width])
{
	//����Mat������ݵ���unchar����
	uchar* p = src.ptr<uchar>(0);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			dst[i][j] = Cpx(p[i * width + j], 0);
		}
	}
}

void Cpx2Mat(Cpx(*src)[width], Mat dst)
{
	//����Mat������ݵ���unchar����
	uchar* p = dst.ptr<uchar>(0);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			double g = sqrt(src[i][j].r * src[i][j].r);
			p[j + i * width] = (uchar)g;
		}
	}
}

void Cpx2MatDouble(Cpx(*src)[width], Mat dst)
{
	//����Mat������ݵ���double����
	double* p = dst.ptr<double>(0);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			double g = sqrt(src[i][j].r * src[i][j].r + src[i][j].i * src[i][j].i);
			g = log(g + 1);  //ת��Ϊ�����߶�
			p[j + i * width] = (double)g;
		}
	}
}

Mat sharpenImage(const cv::Mat& image)
{
	// ����������˹�˲���
	cv::Mat kernel = (cv::Mat_<float>(3, 3) <<
		0, 1, 0,
		1, -4, 1,
		0, 1, 0);

	// ��ͼ����о������
	cv::Mat sharpenedImage;
	cv::filter2D(image, sharpenedImage, CV_8UC3, kernel);

	return sharpenedImage;
}


Mat blurImage(const cv::Mat& image)
{
	// Ӧ�ø�˹�˲���
	cv::Mat blurredImage;
	cv::GaussianBlur(image, blurredImage, cv::Size(5,5), 0);

	return blurredImage;
}
int main()
{
	Mat img = imread("..\\curry.png");
	imshow("img1", img);

	Mat gray = BGR2GRAY(img);
	//imshow("gray", gray);
	//imwrite("gray1.jpg", gray);

	Mat imgRez = Resize(gray);//ת��Ϊ512*512��ͼƬ
	//imshow("imgRez", imgRez);
	//imwrite("imgRez1.jpg", imgRez);

	/*
	Mat sharpenedImage = sharpenImage(img);
	Mat blurredImage = blurImage(img);
	imwrite("sharp1.jpg", sharpenedImage);
	imwrite("blur1.jpg", blurredImage);
	
	imshow("Blurred Image", blurredImage);

	imshow("Sharpened Image", sharpenedImage);
	*/
	Cpx(*src)[width] = new Cpx[height][width];

	Mat2Cpx(imgRez, src);//��imgRezֵת��src

	Cpx(*dst)[width] = new Cpx[height][width];
	int count = 10;


	for (int i = 0; i < 5; i++)
	{
		FFT2D_simd_omp(src, dst, 1);//���任�����dst
	}
	
	auto t1 = Clock::now();
	for (int i = 0; i < count; i++)
	{
		FFT2D_simd_omp(src, dst, 1);//���任�����dst
	}
	auto t2 = Clock::now();
	cout << "2Dfft_ cost : " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e+6/count << " ms.\n";
	
	/*
	Cpx(*dst2)[width] = new Cpx[height][width];
	double t3 = getTickCount();
	FFT2D(dst, dst2, -1);//��任���������dst2
	double t4 = getTickCount();
	
	//double t3t4 = (t4 - t3) / getTickFrequency();
	//std::cout << "IFFT��ʱ: " << t3t4 << "��" << std::endl;

	Mat out2 = Mat::zeros(height, width, CV_8UC1);
	Cpx2Mat(dst2, out2);
	imshow("out2", out2);
	//imwrite("out2.jpg", out2);
	*/

	/*
	Mat out = Mat::zeros(height, width, CV_64F);
	Cpx2MatDouble(dst, out);
	Mat out3 = Mat::zeros(height, width, CV_8UC1);

	AutoScale(out, out3);
	imshow("out3", out3);
	//imwrite("out3.jpg", out3);
	*/

	waitKey(0);
}



/*
һ��
Mat image1(100, 100, CV_8U);
��������������������������������������������ͣ�

	�������������кܶ��֣����õ�Ӧ���У�

	CV_8U��8λ�޷����ͣ�0~255�������Ҷ�ͼ��

	CV_8UC3����ͨ��8λ�޷����ͣ�������ͨ��ָB������G���̣�R���죩����matlab�е�RGB�����෴��

	���ﴴ������ʱδָ�������ֵ������Ĭ��ֵ�Ĵ�СΪ205.

	3.ָ�������С��ָ���������ͣ����ó�ʼֵ��

	Mat image1(100, 100, CV_8U, 100);
��������ĸ����������������������������������ͣ���ʼֵ��
	���ڻҶ�ͼ�񣺿���ֱ�Ӹ�����ʼֵ��Ҳ����ʹ��Scalar������

	Mat image1(100, 100, CV_8U, 100);
Mat image1(100, 100, CV_8U, Scalar(100));
������ͨ��ͼ��ʹ��Scalar������

	Mat image1(100, 100, CV_8UC3, Scalar(100, 100, 100));


	����
	Mat imageROI(image1, Rect(0,0,10,10));  //�������Ȥ����
	����Rect�������ĸ�������Rect��a,b,c,d��:
a������Ȥ������(cols)����㣻
b������Ȥ������(rows)����㣻
c������Ȥ���������(cols)��
d������Ȥ���������(rows)��


*/