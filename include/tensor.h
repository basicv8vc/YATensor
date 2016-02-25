/*！
* \file tensor.h
* \brief 定义了矩阵数据类型
* \author Jia Li
*
*
*
*/


#ifndef YAT_TENSOR_H_
#define YAT_TENSOR_H_
#include <string>
#include <cmath>
#include <algorithm>
#include <cstddef>

typedef unsigned index_t;							//用于Shape每一维度索引
typedef std::size_t size_t;							//用于Shape的大小(元素个数)


namespace yatensor {

	struct cpu{
		const bool isCPU = true;					//单核CPU

	};

	struct gpu {
		const bool isCPU = false;
	};

	template <int ndim>
	struct Shape {									//仅仅保存Tensor的shape，而不保存数据
		const int ndimension = ndim;
		index_t _shape[ndimension];

		inline Shape(void){}
		inline Shape(const Shape<kdim> &sp) {		//这里的形参还是 Shape<ndim> type
			ndimension = kdim;
			for (index_t i = 0; i < kdim; ++i)
				_shape[i] = sp._shape[i];
		}
		int shape()									//返回维度
		{
			return this->ndimension;
		}


		inline index_t &operator[](index_t idx) {	//返回引用, 类似numpy中的视图,
			return _shape[idx];
		}

		inline bool operator==(const Shape<kdim> &sp) {
			for (index_t i = 0; i < kdim; ++i) {
				if (_shape[i] != sp._shape[i])
					return false;
			}
			return true;
		}

		inline bool operator!=(const Shape<kdim> &sp) {
			for (index_t i = 0; i < kdim; ++i) {
				if (_shape[i] != sp._shape[i])
					return true;
			}
			return false;
		}

		inline Shape<1> Flatten1D() {						 //切记对函数内临时变量返回引用！！！都是对形参或者class成员？
			Shape<1> sp1d;
			sp1d.ndimension = 1;
			sp1d._shape[ndimension] = { 0 };
			index_t dims = 1;
			for (index_t i = 0; i < ndimension; ++i)
				dims *= _shape[i];
			sp1d._shape[0] = dims;

			return sp1d;

		}

		inline Shape<2> Flatten2D() {
			Shape<2> sp2d;
			sp2d.ndimension = 2;
			sp2d._shape[ndimension] = { 0,0 };
			sp2d._shape[1] = _shape[ndimension - 1];

			index_t xdims = 1;
			for (index_t i = 0; i < ndimension - 1; ++i)
				xdims *= _shape[i];
			sp2d._shape[0] = xdims;

			return sp2d;
		}


		inline size_t Size() {								//计算Shape中元素个数
			size_t sz= _shape[0];
			for (index_t i = 1; i < ndimension; ++i)
				sz *= _shape[i];

			return sz;
		}

		inline size_t SliceSize(const index_t &start, const index_t &end) { //[start, end)
			size_t sz = _shape[start];
			for (index_t i = start + 1; i < end; ++i)
				sz *= _shape[i];

			return sz;
		}

		template <index_t start, index_t end>				//非类型模板参数
		inline Shape<end-start> SliceShape(){

			index_t kdim = end - start;
			Shape<kdim> sp;
			for (index_t i = 0; i < kdim; ++i)
				sp._shape[i] = 0;



			for (index_t i = start; i < end; ++i)
				sp._shape[i] = _shape[i];

			return sp;
		}

		//-------------------------------------------
		//几个构造Shape的函数
		//-------------------------------------------

		inline Shape<1> Shape1(const index_t &s0) {
			Shape<1> sp;
			sp._shape[0] = s0;

			return sp;
		}

		inline Shape<2> Shape2(const index_t &s0, const index_t &s1) {
			Shape<2> sp;
			sp._shape[0] = s0;
			sp._shape[1] = s1;

			return sp;
		}

		inline Shape<3> Shape3(const index_t &s0, const index_t &s1, const index_t &s2) {
			Shape<3> sp;
			sp._shape[0] = s0;
			sp._shape[1] = s1;
			sp._shape[2] = s2;

			return sp;
		}

		inline Shape<4> Shape4(const index_t &s0, const index_t &s1,
				               const index_t &s2, const index_t &s3) {
			Shape<4> sp;
			sp._shape[0] = s0;
			sp._shape[1] = s1;
			sp._shape[2] = s2;
			sp._shape[3] = s3;

			return sp;
		}

		inline Shape<5> Shape5(const index_t &s0, const index_t &s1,
							   const index_t &s2, const index_t &s3,
					           const index_t &s4) {
			Shape<5> sp;
			sp._shape[0] = s0;
			sp._shape[1] = s1;
			sp._shape[2] = s2;
			sp._shape[3] = s3;
			sp._shape[4] = s4;

			return sp;
		}
























	};


}



















#endif // !YAT_TENSOR_H_
