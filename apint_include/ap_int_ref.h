/************************************************************************************
 *  (c) Copyright 2014-2020 Falcon Computing Solutions, Inc. All rights reserved.
 *
 *  This file contains confidential and proprietary information
 *  of Falcon Computing Solutions, Inc. and is protected under U.S. and
 *  international copyright and other intellectual property laws.
 *
 ************************************************************************************/

#ifndef __AP_INT_REF_H__
#define __AP_INT_REF_H__

#ifndef __cplusplus
#error "C++ is required to include this header file"

#else

template <int _AP_W, bool _AP_S>
struct ap_bit_ref;

template <int _AP_W>
struct ap_bit_ref<_AP_W, true> {
    typedef ap_int<_AP_W> ref_type;

public:
  // copy ctor
  ap_bit_ref(const ap_bit_ref<_AP_W, true>& ref);

  ap_bit_ref(ref_type* bv, int index = 0);

  ap_bit_ref(const ref_type* bv, int index = 0);

  operator bool() const;
  bool to_bool() const;

#define ASSIGN_WITH_CTYPE(_Tp)                          \
  ap_bit_ref& operator=(_Tp val);

  ASSIGN_WITH_CTYPE(bool)
  ASSIGN_WITH_CTYPE(char)
  ASSIGN_WITH_CTYPE(signed char)
  ASSIGN_WITH_CTYPE(unsigned char)
  ASSIGN_WITH_CTYPE(short)
  ASSIGN_WITH_CTYPE(unsigned short)
  ASSIGN_WITH_CTYPE(int)
  ASSIGN_WITH_CTYPE(unsigned int)
  ASSIGN_WITH_CTYPE(long)
  ASSIGN_WITH_CTYPE(unsigned long)
  ASSIGN_WITH_CTYPE(ap_slong)
  ASSIGN_WITH_CTYPE(ap_ulong)

#undef ASSIGN_WITH_CTYPE

#define ASSIGN_WITH_CTYPE_FP(_Tp)                           \
  ap_bit_ref& operator=(_Tp val);

#if _AP_ENABLE_HALF_ == 1
  ASSIGN_WITH_CTYPE_FP(half)
#endif
  ASSIGN_WITH_CTYPE_FP(float)
  ASSIGN_WITH_CTYPE_FP(double)

#undef ASSIGN_WITH_CTYPE_FP


  template <int _AP_W2>
  ap_bit_ref& operator=(const ap_int<_AP_W2>& val);
  template <int _AP_W2>
  ap_bit_ref& operator=(const ap_uint<_AP_W2>& val);

  ap_bit_ref& operator=(const ap_bit_ref& val);

  template <int _AP_W2, bool _AP_S2>
  ap_bit_ref& operator=(const ap_bit_ref<_AP_W2, _AP_S2>& val);

  template <int _AP_W2, bool _AP_S2>
  bool operator==(const ap_bit_ref<_AP_W2, _AP_S2>& op);

  template <int _AP_W2, bool _AP_S2>
  bool operator!=(const ap_bit_ref<_AP_W2, _AP_S2>& op);

  bool get() const;

  bool get();

  template <int _AP_W3>
  void set(const ap_uint<_AP_W3>& val);

  bool operator~() const;

  int length() const;

#ifndef __SYNTHESIS__
#else
  // XXX HLS will delete this in synthesis
  char* to_string() const;
#endif
};

template <int _AP_W>
struct ap_bit_ref<_AP_W, false> {
    typedef ap_uint<_AP_W> ref_type;

public:
  // copy ctor
  ap_bit_ref(const ap_bit_ref<_AP_W, false>& ref);

  ap_bit_ref(ref_type* bv, int index = 0);

  ap_bit_ref(const ref_type* bv, int index = 0);

  operator bool() const;
  bool to_bool() const;

#define ASSIGN_WITH_CTYPE(_Tp)                          \
  ap_bit_ref& operator=(_Tp val);

  ASSIGN_WITH_CTYPE(bool)
  ASSIGN_WITH_CTYPE(char)
  ASSIGN_WITH_CTYPE(signed char)
  ASSIGN_WITH_CTYPE(unsigned char)
  ASSIGN_WITH_CTYPE(short)
  ASSIGN_WITH_CTYPE(unsigned short)
  ASSIGN_WITH_CTYPE(int)
  ASSIGN_WITH_CTYPE(unsigned int)
  ASSIGN_WITH_CTYPE(long)
  ASSIGN_WITH_CTYPE(unsigned long)
  ASSIGN_WITH_CTYPE(ap_slong)
  ASSIGN_WITH_CTYPE(ap_ulong)

#undef ASSIGN_WITH_CTYPE

#define ASSIGN_WITH_CTYPE_FP(_Tp)                           \
  ap_bit_ref& operator=(_Tp val);

#if _AP_ENABLE_HALF_ == 1
  ASSIGN_WITH_CTYPE_FP(half)
#endif
  ASSIGN_WITH_CTYPE_FP(float)
  ASSIGN_WITH_CTYPE_FP(double)

#undef ASSIGN_WITH_CTYPE_FP


  template <int _AP_W2>
  ap_bit_ref& operator=(const ap_int<_AP_W2>& val);
  template <int _AP_W2>
  ap_bit_ref& operator=(const ap_uint<_AP_W2>& val);

  ap_bit_ref& operator=(const ap_bit_ref& val);

  template <int _AP_W2, bool _AP_S2>
  ap_bit_ref& operator=(const ap_bit_ref<_AP_W2, _AP_S2>& val);

  template <int _AP_W2, bool _AP_S2>
  bool operator==(const ap_bit_ref<_AP_W2, _AP_S2>& op);

  template <int _AP_W2, bool _AP_S2>
  bool operator!=(const ap_bit_ref<_AP_W2, _AP_S2>& op);

  bool get() const;

  bool get();

  template <int _AP_W3>
  void set(const ap_uint<_AP_W3>& val);

  bool operator~() const;

  int length() const;

#ifndef __SYNTHESIS__
#else
  // XXX HLS will delete this in synthesis
  char* to_string() const;
#endif
};

#endif // ifndef __cplusplus
#endif // ifndef __AP_INT_REF_H__
