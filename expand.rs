#![feature(prelude_import)]
#![warn(clippy::all, clippy::pedantic, clippy::nursery, clippy::cargo)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::similar_names)]
#![allow(clippy::too_many_arguments)]
#![feature(binary_heap_into_iter_sorted)]
#![feature(associated_type_bounds)]
#![feature(generic_associated_types)]
#![feature(box_syntax)]
#![feature(let_chains)]
#[prelude_import]
use std::prelude::rust_2021::*;
#[macro_use]
extern crate std;
extern crate approx;
extern crate itertools;
extern crate more_asserts;
extern crate ndarray;
extern crate ndarray_linalg;
extern crate ndarray_stats;
extern crate num;
extern crate rand;
extern crate serde;
extern crate shh;
extern crate truncnorm;
pub mod affine {
    #![allow(non_snake_case, clippy::module_name_repetitions)]
    //! Representation of affine transformations
    use crate::bounds::Bounds1;
    use crate::tensorshape::TensorShape;
    use crate::NNVFloat;
    use ndarray::{
        concatenate, iter::Lanes, Array, Array1, Array2, Array4, ArrayView1, ArrayView2,
        ArrayViewMut0, ArrayViewMut1, ArrayViewMut2, Axis, Dimension, Ix1, Ix2, Ix4, IxDyn,
        ShapeError, Zip,
    };
    use serde::{Deserialize, Serialize};
    use std::convert::TryFrom;
    use std::fmt::{Debug, Display};
    use std::ops::{Add, AddAssign, Mul, MulAssign};
    pub type Affine2 = Affine<Ix2>;
    pub type Affine4 = Affine<Ix4>;
    /// Affine map data structure
    pub struct Affine<D: Dimension> {
        basis: Array<NNVFloat, D>,
        shift: Array1<NNVFloat>,
    }
    #[automatically_derived]
    #[allow(unused_qualifications)]
    impl<D: ::core::clone::Clone + Dimension> ::core::clone::Clone for Affine<D> {
        #[inline]
        fn clone(&self) -> Affine<D> {
            match *self {
                Affine {
                    basis: ref __self_0_0,
                    shift: ref __self_0_1,
                } => Affine {
                    basis: ::core::clone::Clone::clone(&(*__self_0_0)),
                    shift: ::core::clone::Clone::clone(&(*__self_0_1)),
                },
            }
        }
    }
    #[automatically_derived]
    #[allow(unused_qualifications)]
    impl<D: ::core::default::Default + Dimension> ::core::default::Default for Affine<D> {
        #[inline]
        fn default() -> Affine<D> {
            Affine {
                basis: ::core::default::Default::default(),
                shift: ::core::default::Default::default(),
            }
        }
    }
    #[automatically_derived]
    #[allow(unused_qualifications)]
    impl<D: ::core::fmt::Debug + Dimension> ::core::fmt::Debug for Affine<D> {
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            match *self {
                Affine {
                    basis: ref __self_0_0,
                    shift: ref __self_0_1,
                } => {
                    let debug_trait_builder =
                        &mut ::core::fmt::Formatter::debug_struct(f, "Affine");
                    let _ = ::core::fmt::DebugStruct::field(
                        debug_trait_builder,
                        "basis",
                        &&(*__self_0_0),
                    );
                    let _ = ::core::fmt::DebugStruct::field(
                        debug_trait_builder,
                        "shift",
                        &&(*__self_0_1),
                    );
                    ::core::fmt::DebugStruct::finish(debug_trait_builder)
                }
            }
        }
    }
    impl<D: Dimension> ::core::marker::StructuralPartialEq for Affine<D> {}
    #[automatically_derived]
    #[allow(unused_qualifications)]
    impl<D: ::core::cmp::PartialEq + Dimension> ::core::cmp::PartialEq for Affine<D> {
        #[inline]
        fn eq(&self, other: &Affine<D>) -> bool {
            match *other {
                Affine {
                    basis: ref __self_1_0,
                    shift: ref __self_1_1,
                } => match *self {
                    Affine {
                        basis: ref __self_0_0,
                        shift: ref __self_0_1,
                    } => (*__self_0_0) == (*__self_1_0) && (*__self_0_1) == (*__self_1_1),
                },
            }
        }
        #[inline]
        fn ne(&self, other: &Affine<D>) -> bool {
            match *other {
                Affine {
                    basis: ref __self_1_0,
                    shift: ref __self_1_1,
                } => match *self {
                    Affine {
                        basis: ref __self_0_0,
                        shift: ref __self_0_1,
                    } => (*__self_0_0) != (*__self_1_0) || (*__self_0_1) != (*__self_1_1),
                },
            }
        }
    }
    #[doc(hidden)]
    #[allow(non_upper_case_globals, unused_attributes, unused_qualifications)]
    const _: () = {
        #[allow(unused_extern_crates, clippy::useless_attribute)]
        extern crate serde as _serde;
        #[automatically_derived]
        impl<'de, D: Dimension> _serde::Deserialize<'de> for Affine<D>
        where
            D: _serde::Deserialize<'de>,
        {
            fn deserialize<__D>(__deserializer: __D) -> _serde::__private::Result<Self, __D::Error>
            where
                __D: _serde::Deserializer<'de>,
            {
                #[allow(non_camel_case_types)]
                enum __Field {
                    __field0,
                    __field1,
                    __ignore,
                }
                struct __FieldVisitor;
                impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                    type Value = __Field;
                    fn expecting(
                        &self,
                        __formatter: &mut _serde::__private::Formatter,
                    ) -> _serde::__private::fmt::Result {
                        _serde::__private::Formatter::write_str(__formatter, "field identifier")
                    }
                    fn visit_u64<__E>(
                        self,
                        __value: u64,
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            0u64 => _serde::__private::Ok(__Field::__field0),
                            1u64 => _serde::__private::Ok(__Field::__field1),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                    fn visit_str<__E>(
                        self,
                        __value: &str,
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            "basis" => _serde::__private::Ok(__Field::__field0),
                            "shift" => _serde::__private::Ok(__Field::__field1),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                    fn visit_bytes<__E>(
                        self,
                        __value: &[u8],
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            b"basis" => _serde::__private::Ok(__Field::__field0),
                            b"shift" => _serde::__private::Ok(__Field::__field1),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                }
                impl<'de> _serde::Deserialize<'de> for __Field {
                    #[inline]
                    fn deserialize<__D>(
                        __deserializer: __D,
                    ) -> _serde::__private::Result<Self, __D::Error>
                    where
                        __D: _serde::Deserializer<'de>,
                    {
                        _serde::Deserializer::deserialize_identifier(__deserializer, __FieldVisitor)
                    }
                }
                struct __Visitor<'de, D: Dimension>
                where
                    D: _serde::Deserialize<'de>,
                {
                    marker: _serde::__private::PhantomData<Affine<D>>,
                    lifetime: _serde::__private::PhantomData<&'de ()>,
                }
                impl<'de, D: Dimension> _serde::de::Visitor<'de> for __Visitor<'de, D>
                where
                    D: _serde::Deserialize<'de>,
                {
                    type Value = Affine<D>;
                    fn expecting(
                        &self,
                        __formatter: &mut _serde::__private::Formatter,
                    ) -> _serde::__private::fmt::Result {
                        _serde::__private::Formatter::write_str(__formatter, "struct Affine")
                    }
                    #[inline]
                    fn visit_seq<__A>(
                        self,
                        mut __seq: __A,
                    ) -> _serde::__private::Result<Self::Value, __A::Error>
                    where
                        __A: _serde::de::SeqAccess<'de>,
                    {
                        let __field0 = match match _serde::de::SeqAccess::next_element::<
                            Array<NNVFloat, D>,
                        >(&mut __seq)
                        {
                            _serde::__private::Ok(__val) => __val,
                            _serde::__private::Err(__err) => {
                                return _serde::__private::Err(__err);
                            }
                        } {
                            _serde::__private::Some(__value) => __value,
                            _serde::__private::None => {
                                return _serde::__private::Err(_serde::de::Error::invalid_length(
                                    0usize,
                                    &"struct Affine with 2 elements",
                                ));
                            }
                        };
                        let __field1 = match match _serde::de::SeqAccess::next_element::<
                            Array1<NNVFloat>,
                        >(&mut __seq)
                        {
                            _serde::__private::Ok(__val) => __val,
                            _serde::__private::Err(__err) => {
                                return _serde::__private::Err(__err);
                            }
                        } {
                            _serde::__private::Some(__value) => __value,
                            _serde::__private::None => {
                                return _serde::__private::Err(_serde::de::Error::invalid_length(
                                    1usize,
                                    &"struct Affine with 2 elements",
                                ));
                            }
                        };
                        _serde::__private::Ok(Affine {
                            basis: __field0,
                            shift: __field1,
                        })
                    }
                    #[inline]
                    fn visit_map<__A>(
                        self,
                        mut __map: __A,
                    ) -> _serde::__private::Result<Self::Value, __A::Error>
                    where
                        __A: _serde::de::MapAccess<'de>,
                    {
                        let mut __field0: _serde::__private::Option<Array<NNVFloat, D>> =
                            _serde::__private::None;
                        let mut __field1: _serde::__private::Option<Array1<NNVFloat>> =
                            _serde::__private::None;
                        while let _serde::__private::Some(__key) =
                            match _serde::de::MapAccess::next_key::<__Field>(&mut __map) {
                                _serde::__private::Ok(__val) => __val,
                                _serde::__private::Err(__err) => {
                                    return _serde::__private::Err(__err);
                                }
                            }
                        {
                            match __key {
                                __Field::__field0 => {
                                    if _serde::__private::Option::is_some(&__field0) {
                                        return _serde::__private::Err(
                                            <__A::Error as _serde::de::Error>::duplicate_field(
                                                "basis",
                                            ),
                                        );
                                    }
                                    __field0 = _serde::__private::Some(
                                        match _serde::de::MapAccess::next_value::<Array<NNVFloat, D>>(
                                            &mut __map,
                                        ) {
                                            _serde::__private::Ok(__val) => __val,
                                            _serde::__private::Err(__err) => {
                                                return _serde::__private::Err(__err);
                                            }
                                        },
                                    );
                                }
                                __Field::__field1 => {
                                    if _serde::__private::Option::is_some(&__field1) {
                                        return _serde::__private::Err(
                                            <__A::Error as _serde::de::Error>::duplicate_field(
                                                "shift",
                                            ),
                                        );
                                    }
                                    __field1 = _serde::__private::Some(
                                        match _serde::de::MapAccess::next_value::<Array1<NNVFloat>>(
                                            &mut __map,
                                        ) {
                                            _serde::__private::Ok(__val) => __val,
                                            _serde::__private::Err(__err) => {
                                                return _serde::__private::Err(__err);
                                            }
                                        },
                                    );
                                }
                                _ => {
                                    let _ = match _serde::de::MapAccess::next_value::<
                                        _serde::de::IgnoredAny,
                                    >(&mut __map)
                                    {
                                        _serde::__private::Ok(__val) => __val,
                                        _serde::__private::Err(__err) => {
                                            return _serde::__private::Err(__err);
                                        }
                                    };
                                }
                            }
                        }
                        let __field0 = match __field0 {
                            _serde::__private::Some(__field0) => __field0,
                            _serde::__private::None => {
                                match _serde::__private::de::missing_field("basis") {
                                    _serde::__private::Ok(__val) => __val,
                                    _serde::__private::Err(__err) => {
                                        return _serde::__private::Err(__err);
                                    }
                                }
                            }
                        };
                        let __field1 = match __field1 {
                            _serde::__private::Some(__field1) => __field1,
                            _serde::__private::None => {
                                match _serde::__private::de::missing_field("shift") {
                                    _serde::__private::Ok(__val) => __val,
                                    _serde::__private::Err(__err) => {
                                        return _serde::__private::Err(__err);
                                    }
                                }
                            }
                        };
                        _serde::__private::Ok(Affine {
                            basis: __field0,
                            shift: __field1,
                        })
                    }
                }
                const FIELDS: &'static [&'static str] = &["basis", "shift"];
                _serde::Deserializer::deserialize_struct(
                    __deserializer,
                    "Affine",
                    FIELDS,
                    __Visitor {
                        marker: _serde::__private::PhantomData::<Affine<D>>,
                        lifetime: _serde::__private::PhantomData,
                    },
                )
            }
        }
    };
    #[doc(hidden)]
    #[allow(non_upper_case_globals, unused_attributes, unused_qualifications)]
    const _: () = {
        #[allow(unused_extern_crates, clippy::useless_attribute)]
        extern crate serde as _serde;
        #[automatically_derived]
        impl<D: Dimension> _serde::Serialize for Affine<D>
        where
            D: _serde::Serialize,
        {
            fn serialize<__S>(
                &self,
                __serializer: __S,
            ) -> _serde::__private::Result<__S::Ok, __S::Error>
            where
                __S: _serde::Serializer,
            {
                let mut __serde_state = match _serde::Serializer::serialize_struct(
                    __serializer,
                    "Affine",
                    false as usize + 1 + 1,
                ) {
                    _serde::__private::Ok(__val) => __val,
                    _serde::__private::Err(__err) => {
                        return _serde::__private::Err(__err);
                    }
                };
                match _serde::ser::SerializeStruct::serialize_field(
                    &mut __serde_state,
                    "basis",
                    &self.basis,
                ) {
                    _serde::__private::Ok(__val) => __val,
                    _serde::__private::Err(__err) => {
                        return _serde::__private::Err(__err);
                    }
                };
                match _serde::ser::SerializeStruct::serialize_field(
                    &mut __serde_state,
                    "shift",
                    &self.shift,
                ) {
                    _serde::__private::Ok(__val) => __val,
                    _serde::__private::Err(__err) => {
                        return _serde::__private::Err(__err);
                    }
                };
                _serde::ser::SerializeStruct::end(__serde_state)
            }
        }
    };
    impl<D: Dimension> Display for Affine<D> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::result::Result<(), std::fmt::Error> {
            f.write_fmt(::core::fmt::Arguments::new_v1(
                &["Basis ", " Shift "],
                &[
                    ::core::fmt::ArgumentV1::new_debug(&self.basis.shape()),
                    ::core::fmt::ArgumentV1::new_debug(&self.shift.shape()),
                ],
            ))
        }
    }
    impl<D: Dimension> Affine<D> {
        pub fn ndim(&self) -> usize {
            self.basis.ndim()
        }
        pub fn shift(&self) -> ArrayView1<NNVFloat> {
            self.shift.view()
        }
        pub fn shift_mut(&mut self) -> ArrayViewMut1<NNVFloat> {
            self.shift.view_mut()
        }
        pub fn into_dyn(self) -> Affine<IxDyn> {
            Affine {
                basis: self.basis.into_dyn(),
                shift: self.shift,
            }
        }
    }
    impl<D: Dimension + ndarray::RemoveAxis> Affine<D> {
        /// Get a single equation (i.e., a set of coefficients and a shift/RHS)
        ///
        /// # Panics
        #[must_use]
        pub fn get_eqn(&self, index: usize) -> Self {
            let idx = isize::try_from(index).unwrap();
            let basis = self
                .basis
                .slice_axis(Axis(0), ndarray::Slice::new(idx, Some(idx + 1), 1))
                .to_owned();
            let shift = self
                .shift
                .index_axis(Axis(0), index)
                .to_owned()
                .insert_axis(Axis(0));
            Self { basis, shift }
        }
    }
    impl Affine<IxDyn> {
        /// # Errors
        pub fn into_dimensionality<D: Dimension>(self) -> Result<Affine<D>, ShapeError> {
            let shift = self.shift;
            self.basis
                .into_dimensionality::<D>()
                .map(|basis| Affine { basis, shift })
        }
    }
    /// Assumes that the affine is f(x) = Ax + b
    impl Affine2 {
        /// # Panics
        /// If improper shapes are passed in
        pub fn new(basis: Array2<NNVFloat>, shift: Array1<NNVFloat>) -> Self {
            if true {
                match (&basis.shape()[0], &shift.len()) {
                    (left_val, right_val) => {
                        if !(*left_val == *right_val) {
                            let kind = ::core::panicking::AssertKind::Eq;
                            ::core::panicking::assert_failed(
                                kind,
                                &*left_val,
                                &*right_val,
                                ::core::option::Option::None,
                            );
                        }
                    }
                };
            };
            Self { basis, shift }
        }
        pub fn identity(ndim: usize) -> Self {
            Self {
                basis: Array2::eye(ndim),
                shift: Array1::zeros(ndim),
            }
        }
        pub fn basis(&self) -> ArrayView2<NNVFloat> {
            self.basis.view()
        }
        pub fn basis_mut(&mut self) -> ArrayViewMut2<NNVFloat> {
            self.basis.view_mut()
        }
        pub fn input_dim(&self) -> usize {
            self.basis.shape()[1]
        }
        pub fn output_dim(&self) -> usize {
            self.shift.len()
        }
        pub fn shape(&self) -> &[usize] {
            self.basis.shape()
        }
        pub fn zero_eqn(&mut self, idx: usize) {
            self.basis.index_axis_mut(Axis(0), idx).fill(num::zero());
            self.shift.index_axis_mut(Axis(0), idx).fill(num::zero());
        }
        pub fn get_raw_augmented(&self) -> Array2<NNVFloat> {
            ::ndarray::concatenate(
                Axis(1),
                &[
                    ::ndarray::ArrayView::from(&self.basis),
                    ::ndarray::ArrayView::from(&self.shift.clone().insert_axis(Axis(0))),
                ],
            )
            .unwrap()
        }
        pub fn get_eqn_mut(
            &mut self,
            index: usize,
        ) -> (ArrayViewMut1<NNVFloat>, ArrayViewMut0<NNVFloat>) {
            (
                self.basis.index_axis_mut(Axis(0), index),
                self.shift.index_axis_mut(Axis(0), index),
            )
        }
        pub fn vars(&self) -> Lanes<NNVFloat, Ix1> {
            self.basis.columns()
        }
        pub fn apply(&self, x: &ArrayView1<NNVFloat>) -> Array1<NNVFloat> {
            self.basis.dot(x) + &self.shift
        }
        pub fn apply_matrix(&self, x: &ArrayView2<NNVFloat>) -> Array2<NNVFloat> {
            &self.basis.dot(x) + &self.shift.view().insert_axis(Axis(1))
        }
        pub fn split_at(&self, index: usize) -> (Self, Self) {
            let (basis_head, basis_tail) = self.basis.view().split_at(Axis(1), index);
            (
                Self {
                    basis: basis_head.to_owned(),
                    shift: self.shift.clone(),
                },
                Self {
                    basis: basis_tail.to_owned(),
                    shift: self.shift.clone(),
                },
            )
        }
        /// # Panics
        #[must_use]
        pub fn append(mut self, other: &Self) -> Self {
            self.basis.append(Axis(1), other.basis.view()).unwrap();
            self
        }
    }
    impl Affine2 {
        pub fn signed_apply(&self, bounds: &Bounds1) -> Bounds1 {
            let lower = crate::util::signed_dot(
                &self.basis.view(),
                &bounds.lower().view(),
                &bounds.upper().view(),
            ) + &self.shift;
            let upper = crate::util::signed_dot(
                &self.basis.view(),
                &bounds.upper().view(),
                &bounds.lower().view(),
            ) + &self.shift;
            Bounds1::new(lower.view(), upper.view())
        }
        /// # Panics
        #[must_use]
        pub fn signed_compose(&self, pos_rhs: &Self, neg_rhs: &Self) -> Self {
            if true {
                match (&self.input_dim(), &pos_rhs.output_dim()) {
                    (left_val, right_val) => {
                        if !(*left_val == *right_val) {
                            let kind = ::core::panicking::AssertKind::Eq;
                            ::core::panicking::assert_failed(
                                kind,
                                &*left_val,
                                &*right_val,
                                ::core::option::Option::Some(::core::fmt::Arguments::new_v1(
                                    &["self input dim: ", ", pos_rhs output dim: "],
                                    &[
                                        ::core::fmt::ArgumentV1::new_display(&self.input_dim()),
                                        ::core::fmt::ArgumentV1::new_display(&pos_rhs.output_dim()),
                                    ],
                                )),
                            );
                        }
                    }
                };
            };
            if true {
                match (&self.input_dim(), &neg_rhs.output_dim()) {
                    (left_val, right_val) => {
                        if !(*left_val == *right_val) {
                            let kind = ::core::panicking::AssertKind::Eq;
                            ::core::panicking::assert_failed(
                                kind,
                                &*left_val,
                                &*right_val,
                                ::core::option::Option::None,
                            );
                        }
                    }
                };
            };
            Self {
                basis: crate::util::signed_matmul(
                    &self.basis.view(),
                    &pos_rhs.basis.view(),
                    &neg_rhs.basis.view(),
                ),
                shift: &crate::util::signed_dot(
                    &self.basis.view(),
                    &pos_rhs.shift.view(),
                    &neg_rhs.shift.view(),
                ) + &self.shift,
            }
        }
    }
    impl Affine2 {
        /// # Panics
        pub fn scale_eqns(&mut self, x: ArrayView1<NNVFloat>) {
            if true {
                match (&self.basis.nrows(), &x.len()) {
                    (left_val, right_val) => {
                        if !(*left_val == *right_val) {
                            let kind = ::core::panicking::AssertKind::Eq;
                            ::core::panicking::assert_failed(
                                kind,
                                &*left_val,
                                &*right_val,
                                ::core::option::Option::None,
                            );
                        }
                    }
                };
            };
            Zip::from(self.basis.rows_mut())
                .and(self.shift.view_mut())
                .and(x)
                .for_each(|mut row, shift, &x| {
                    row.assign(&(&row * x));
                    *shift *= x;
                });
        }
    }
    /// Add scalar
    impl<D: Dimension> Add<NNVFloat> for Affine<D> {
        type Output = Self;
        fn add(self, rhs: NNVFloat) -> Self {
            Self {
                basis: self.basis,
                shift: &self.shift + rhs,
            }
        }
    }
    /// Add vec
    impl<D: Dimension> Add<Array1<NNVFloat>> for Affine<D> {
        type Output = Self;
        fn add(self, rhs: Array1<NNVFloat>) -> Self {
            Self {
                basis: self.basis,
                shift: &self.shift + rhs,
            }
        }
    }
    impl<D: Dimension> AddAssign<NNVFloat> for Affine<D> {
        fn add_assign(&mut self, rhs: NNVFloat) {
            self.shift += rhs;
        }
    }
    /// Scale Affine by scalar
    impl<D: Dimension> Mul<NNVFloat> for Affine<D> {
        type Output = Self;
        fn mul(self, rhs: NNVFloat) -> Self {
            Self {
                basis: &self.basis * rhs,
                shift: &self.shift * rhs,
            }
        }
    }
    /// Scale Affine by vector
    impl Mul<Array1<NNVFloat>> for Affine2 {
        type Output = Self;
        fn mul(self, rhs: Array1<NNVFloat>) -> Self {
            Self {
                basis: &self.basis * &rhs,
                shift: &self.shift * rhs,
            }
        }
    }
    /// Scale Affine by vector
    impl MulAssign<Array1<NNVFloat>> for Affine2 {
        fn mul_assign(&mut self, rhs: Array1<NNVFloat>) {
            self.basis *= &rhs;
            self.shift *= &rhs;
        }
    }
    /// Scale Affine by scalar
    impl<D: Dimension> MulAssign<NNVFloat> for Affine<D> {
        fn mul_assign(&mut self, scalar: NNVFloat) {
            self.basis *= scalar;
            self.shift *= scalar;
        }
    }
    impl<'a, 'b> Mul<&'b Affine2> for &'a Affine2 {
        type Output = Affine2;
        #[allow(clippy::suspicious_arithmetic_impl)]
        fn mul(self, rhs: &'b Affine2) -> Affine2 {
            let basis = self.basis.dot(&rhs.basis);
            let shift = self.basis.dot(&rhs.shift) + self.shift.clone();
            Affine { basis, shift }
        }
    }
    /// Apply Affine to Affine
    impl Mul<&Self> for Affine2 {
        type Output = Self;
        #[allow(clippy::suspicious_arithmetic_impl)]
        fn mul(self, rhs: &Self) -> Self {
            let basis = self.basis.dot(&rhs.basis);
            let shift = self.basis.dot(&rhs.shift) + self.shift;
            Self { basis, shift }
        }
    }
    impl Affine<Ix4> {
        pub const fn new(basis: Array4<NNVFloat>, shift: Array1<NNVFloat>) -> Self {
            Self { basis, shift }
        }
        pub fn output_channels(&self) -> usize {
            self.shift.len()
        }
        pub fn input_shape(&self) -> TensorShape {
            TensorShape::new(<[_]>::into_vec(box [
                None,
                None,
                Some(self.basis.shape()[2]),
            ]))
        }
    }
}
pub mod bounds {
    #![allow(clippy::module_name_repetitions)]
    use crate::affine::Affine2;
    use crate::rand::distributions::Distribution;
    use crate::rand::SeedableRng;
    use crate::NNVFloat;
    use ndarray::iter::{Lanes, LanesMut};
    use ndarray::Array2;
    use ndarray::ArrayView1;
    use ndarray::Axis;
    use ndarray::Ix2;
    use ndarray::RemoveAxis;
    use ndarray::Zip;
    use ndarray::{concatenate, Array1};
    use ndarray::{stack, Array, Dimension};
    use ndarray::{ArrayView, ArrayViewMut, ArrayViewMut1};
    use num::Float;
    use num::Zero;
    use ordered_float::OrderedFloat;
    use rand::distributions::Uniform;
    use rand::rngs::StdRng;
    use serde::{Deserialize, Serialize};
    use std::cmp;
    use std::fmt::Display;
    use std::ops::{Mul, MulAssign};
    pub type Bounds1 = Bounds<Ix2>;
    pub struct Bounds<D: Dimension> {
        data: Array<NNVFloat, D>,
    }
    #[automatically_derived]
    #[allow(unused_qualifications)]
    impl<D: ::core::clone::Clone + Dimension> ::core::clone::Clone for Bounds<D> {
        #[inline]
        fn clone(&self) -> Bounds<D> {
            match *self {
                Bounds {
                    data: ref __self_0_0,
                } => Bounds {
                    data: ::core::clone::Clone::clone(&(*__self_0_0)),
                },
            }
        }
    }
    #[automatically_derived]
    #[allow(unused_qualifications)]
    impl<D: ::core::default::Default + Dimension> ::core::default::Default for Bounds<D> {
        #[inline]
        fn default() -> Bounds<D> {
            Bounds {
                data: ::core::default::Default::default(),
            }
        }
    }
    #[automatically_derived]
    #[allow(unused_qualifications)]
    impl<D: ::core::fmt::Debug + Dimension> ::core::fmt::Debug for Bounds<D> {
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            match *self {
                Bounds {
                    data: ref __self_0_0,
                } => {
                    let debug_trait_builder =
                        &mut ::core::fmt::Formatter::debug_struct(f, "Bounds");
                    let _ = ::core::fmt::DebugStruct::field(
                        debug_trait_builder,
                        "data",
                        &&(*__self_0_0),
                    );
                    ::core::fmt::DebugStruct::finish(debug_trait_builder)
                }
            }
        }
    }
    impl<D: Dimension> ::core::marker::StructuralPartialEq for Bounds<D> {}
    #[automatically_derived]
    #[allow(unused_qualifications)]
    impl<D: ::core::cmp::PartialEq + Dimension> ::core::cmp::PartialEq for Bounds<D> {
        #[inline]
        fn eq(&self, other: &Bounds<D>) -> bool {
            match *other {
                Bounds {
                    data: ref __self_1_0,
                } => match *self {
                    Bounds {
                        data: ref __self_0_0,
                    } => (*__self_0_0) == (*__self_1_0),
                },
            }
        }
        #[inline]
        fn ne(&self, other: &Bounds<D>) -> bool {
            match *other {
                Bounds {
                    data: ref __self_1_0,
                } => match *self {
                    Bounds {
                        data: ref __self_0_0,
                    } => (*__self_0_0) != (*__self_1_0),
                },
            }
        }
    }
    #[doc(hidden)]
    #[allow(non_upper_case_globals, unused_attributes, unused_qualifications)]
    const _: () = {
        #[allow(unused_extern_crates, clippy::useless_attribute)]
        extern crate serde as _serde;
        #[automatically_derived]
        impl<'de, D: Dimension> _serde::Deserialize<'de> for Bounds<D>
        where
            D: _serde::Deserialize<'de>,
        {
            fn deserialize<__D>(__deserializer: __D) -> _serde::__private::Result<Self, __D::Error>
            where
                __D: _serde::Deserializer<'de>,
            {
                #[allow(non_camel_case_types)]
                enum __Field {
                    __field0,
                    __ignore,
                }
                struct __FieldVisitor;
                impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                    type Value = __Field;
                    fn expecting(
                        &self,
                        __formatter: &mut _serde::__private::Formatter,
                    ) -> _serde::__private::fmt::Result {
                        _serde::__private::Formatter::write_str(__formatter, "field identifier")
                    }
                    fn visit_u64<__E>(
                        self,
                        __value: u64,
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            0u64 => _serde::__private::Ok(__Field::__field0),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                    fn visit_str<__E>(
                        self,
                        __value: &str,
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            "data" => _serde::__private::Ok(__Field::__field0),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                    fn visit_bytes<__E>(
                        self,
                        __value: &[u8],
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            b"data" => _serde::__private::Ok(__Field::__field0),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                }
                impl<'de> _serde::Deserialize<'de> for __Field {
                    #[inline]
                    fn deserialize<__D>(
                        __deserializer: __D,
                    ) -> _serde::__private::Result<Self, __D::Error>
                    where
                        __D: _serde::Deserializer<'de>,
                    {
                        _serde::Deserializer::deserialize_identifier(__deserializer, __FieldVisitor)
                    }
                }
                struct __Visitor<'de, D: Dimension>
                where
                    D: _serde::Deserialize<'de>,
                {
                    marker: _serde::__private::PhantomData<Bounds<D>>,
                    lifetime: _serde::__private::PhantomData<&'de ()>,
                }
                impl<'de, D: Dimension> _serde::de::Visitor<'de> for __Visitor<'de, D>
                where
                    D: _serde::Deserialize<'de>,
                {
                    type Value = Bounds<D>;
                    fn expecting(
                        &self,
                        __formatter: &mut _serde::__private::Formatter,
                    ) -> _serde::__private::fmt::Result {
                        _serde::__private::Formatter::write_str(__formatter, "struct Bounds")
                    }
                    #[inline]
                    fn visit_seq<__A>(
                        self,
                        mut __seq: __A,
                    ) -> _serde::__private::Result<Self::Value, __A::Error>
                    where
                        __A: _serde::de::SeqAccess<'de>,
                    {
                        let __field0 = match match _serde::de::SeqAccess::next_element::<
                            Array<NNVFloat, D>,
                        >(&mut __seq)
                        {
                            _serde::__private::Ok(__val) => __val,
                            _serde::__private::Err(__err) => {
                                return _serde::__private::Err(__err);
                            }
                        } {
                            _serde::__private::Some(__value) => __value,
                            _serde::__private::None => {
                                return _serde::__private::Err(_serde::de::Error::invalid_length(
                                    0usize,
                                    &"struct Bounds with 1 element",
                                ));
                            }
                        };
                        _serde::__private::Ok(Bounds { data: __field0 })
                    }
                    #[inline]
                    fn visit_map<__A>(
                        self,
                        mut __map: __A,
                    ) -> _serde::__private::Result<Self::Value, __A::Error>
                    where
                        __A: _serde::de::MapAccess<'de>,
                    {
                        let mut __field0: _serde::__private::Option<Array<NNVFloat, D>> =
                            _serde::__private::None;
                        while let _serde::__private::Some(__key) =
                            match _serde::de::MapAccess::next_key::<__Field>(&mut __map) {
                                _serde::__private::Ok(__val) => __val,
                                _serde::__private::Err(__err) => {
                                    return _serde::__private::Err(__err);
                                }
                            }
                        {
                            match __key {
                                __Field::__field0 => {
                                    if _serde::__private::Option::is_some(&__field0) {
                                        return _serde::__private::Err(
                                            <__A::Error as _serde::de::Error>::duplicate_field(
                                                "data",
                                            ),
                                        );
                                    }
                                    __field0 = _serde::__private::Some(
                                        match _serde::de::MapAccess::next_value::<Array<NNVFloat, D>>(
                                            &mut __map,
                                        ) {
                                            _serde::__private::Ok(__val) => __val,
                                            _serde::__private::Err(__err) => {
                                                return _serde::__private::Err(__err);
                                            }
                                        },
                                    );
                                }
                                _ => {
                                    let _ = match _serde::de::MapAccess::next_value::<
                                        _serde::de::IgnoredAny,
                                    >(&mut __map)
                                    {
                                        _serde::__private::Ok(__val) => __val,
                                        _serde::__private::Err(__err) => {
                                            return _serde::__private::Err(__err);
                                        }
                                    };
                                }
                            }
                        }
                        let __field0 = match __field0 {
                            _serde::__private::Some(__field0) => __field0,
                            _serde::__private::None => {
                                match _serde::__private::de::missing_field("data") {
                                    _serde::__private::Ok(__val) => __val,
                                    _serde::__private::Err(__err) => {
                                        return _serde::__private::Err(__err);
                                    }
                                }
                            }
                        };
                        _serde::__private::Ok(Bounds { data: __field0 })
                    }
                }
                const FIELDS: &'static [&'static str] = &["data"];
                _serde::Deserializer::deserialize_struct(
                    __deserializer,
                    "Bounds",
                    FIELDS,
                    __Visitor {
                        marker: _serde::__private::PhantomData::<Bounds<D>>,
                        lifetime: _serde::__private::PhantomData,
                    },
                )
            }
        }
    };
    #[doc(hidden)]
    #[allow(non_upper_case_globals, unused_attributes, unused_qualifications)]
    const _: () = {
        #[allow(unused_extern_crates, clippy::useless_attribute)]
        extern crate serde as _serde;
        #[automatically_derived]
        impl<D: Dimension> _serde::Serialize for Bounds<D>
        where
            D: _serde::Serialize,
        {
            fn serialize<__S>(
                &self,
                __serializer: __S,
            ) -> _serde::__private::Result<__S::Ok, __S::Error>
            where
                __S: _serde::Serializer,
            {
                let mut __serde_state = match _serde::Serializer::serialize_struct(
                    __serializer,
                    "Bounds",
                    false as usize + 1,
                ) {
                    _serde::__private::Ok(__val) => __val,
                    _serde::__private::Err(__err) => {
                        return _serde::__private::Err(__err);
                    }
                };
                match _serde::ser::SerializeStruct::serialize_field(
                    &mut __serde_state,
                    "data",
                    &self.data,
                ) {
                    _serde::__private::Ok(__val) => __val,
                    _serde::__private::Err(__err) => {
                        return _serde::__private::Err(__err);
                    }
                };
                _serde::ser::SerializeStruct::end(__serde_state)
            }
        }
    };
    impl<D: Dimension + ndarray::RemoveAxis> Bounds<D> {
        /// # Panics
        pub fn new<'a, S: Dimension + Dimension<Larger = D>>(
            lower: ArrayView<'a, NNVFloat, S>,
            upper: ArrayView<'a, NNVFloat, S>,
        ) -> Self {
            if true {
                if !Zip::from(&lower).and(&upper).all(|&l, &u| l <= u) {
                    ::core::panicking::panic_fmt(::core::fmt::Arguments::new_v1(
                        &["Input bounds are flipped!"],
                        &[],
                    ))
                };
            };
            let data: Array<NNVFloat, D> = stack(Axis(0), &[lower, upper]).unwrap();
            Self { data }
        }
        pub fn fixed_idxs(&self) -> Array<bool, D::Smaller> {
            Zip::from(self.lower())
                .and(self.upper())
                .map_collect(|&lb, &ub| lb == ub)
        }
        pub fn fixed_vals_or_zeros(&self) -> Array<NNVFloat, D::Smaller> {
            Zip::from(self.lower())
                .and(self.upper())
                .map_collect(|&lb, &ub| if lb == ub { lb } else { NNVFloat::zero() })
        }
        pub fn fixed_vals_or_none(&self) -> Array<Option<NNVFloat>, D::Smaller> {
            Zip::from(self.lower())
                .and(self.upper())
                .map_collect(|&lb, &ub| if lb == ub { Some(lb) } else { None })
        }
        pub fn is_all_finite(&self) -> bool {
            self.data.iter().all(|&x| NNVFloat::is_finite(x))
        }
        pub fn as_tuple(&self) -> (Array<NNVFloat, D::Smaller>, Array<NNVFloat, D::Smaller>) {
            (self.lower().to_owned(), self.upper().to_owned())
        }
        pub fn lower(&self) -> ArrayView<NNVFloat, D::Smaller> {
            self.data.index_axis(Axis(0), 0)
        }
        pub fn lower_mut(&mut self) -> ArrayViewMut<NNVFloat, D::Smaller> {
            self.data.index_axis_mut(Axis(0), 0)
        }
        pub fn upper(&self) -> ArrayView<NNVFloat, D::Smaller> {
            self.data.index_axis(Axis(0), 1)
        }
        pub fn upper_mut(&mut self) -> ArrayViewMut<NNVFloat, D::Smaller> {
            self.data.index_axis_mut(Axis(0), 1)
        }
        pub fn ndim(&self) -> usize {
            self.data.shape().iter().skip(1).product()
        }
        pub fn bounds_iter(&self) -> Lanes<NNVFloat, D::Smaller> {
            self.data.lanes(Axis(0))
        }
        pub fn bounds_iter_mut(&mut self) -> LanesMut<NNVFloat, D::Smaller> {
            self.data.lanes_mut(Axis(0))
        }
        pub fn is_member(&self, x: &ArrayView<NNVFloat, D::Smaller>) -> bool {
            let eps = 1e-5;
            Zip::from(x)
                .and(self.bounds_iter())
                .all(|&x, bounds| bounds[0] - eps <= x && x <= bounds[1] + eps)
        }
        #[must_use]
        pub fn intersect(&self, other: &Self) -> Self {
            let mut intersection = Self {
                data: self.data.clone(),
            };
            Zip::from(self.lower())
                .and(other.lower())
                .map_assign_into(intersection.lower_mut(), |&x, &y| {
                    cmp::max(OrderedFloat(x), OrderedFloat(y)).0
                });
            Zip::from(self.upper())
                .and(other.upper())
                .map_assign_into(intersection.upper_mut(), |&x, &y| {
                    cmp::min(OrderedFloat(x), OrderedFloat(y)).0
                });
            intersection
        }
    }
    impl<D: Dimension + ndarray::RemoveAxis> Bounds<D> {
        /// # Panics
        pub fn subset(&self, rhs: &Self) -> bool {
            Zip::from(self.bounds_iter())
                .and(rhs.bounds_iter())
                .all(|me, rhs| {
                    let diff = me.to_owned() - rhs;
                    let eps = <NNVFloat as num::NumCast>::from(1e-8).unwrap();
                    (diff[[0]] >= NNVFloat::zero() || diff[[0]] <= eps)
                        && (diff[[1]] <= NNVFloat::zero() || diff[[1]] <= eps)
                })
        }
    }
    impl<D: Dimension + ndarray::RemoveAxis> Bounds<D> {
        pub fn sample_uniform(&self, seed: u64) -> Array<NNVFloat, D::Smaller> {
            let mut rng = StdRng::seed_from_u64(seed);
            Zip::from(self.bounds_iter())
                .map_collect(|x| Uniform::new_inclusive(x[0], x[1]).sample(&mut rng))
        }
    }
    impl Bounds1 {
        /// # Panics
        pub fn new_by_dim(dim_bounds: &[ArrayView1<NNVFloat>]) -> Self {
            let dims: Vec<_> = dim_bounds.iter().map(|x| x.insert_axis(Axis(1))).collect();
            Self {
                data: concatenate(Axis(1), &dims).unwrap(),
            }
        }
        pub fn default(dim: usize) -> Self {
            Self {
                data: Array2::default([2, dim]),
            }
        }
        pub fn trivial(dim: usize) -> Self {
            Self::new(
                Array::from_elem(dim, NNVFloat::neg_infinity()).view(),
                Array::from_elem(dim, NNVFloat::infinity()).view(),
            )
        }
        #[must_use]
        pub fn affine_map(&self, aff: &Affine2) -> Self {
            let lower = aff.apply(&self.lower());
            let upper = aff.apply(&self.upper());
            Self::new(lower.view(), upper.view())
        }
        pub fn split_at(&self, index: usize) -> (Self, Self) {
            let (head, tail) = self.data.view().split_at(Axis(1), index);
            (
                Self {
                    data: head.to_owned(),
                },
                Self {
                    data: tail.to_owned(),
                },
            )
        }
        /// # Panics
        #[must_use]
        pub fn append(mut self, other: &Self) -> Self {
            self.data.append(Axis(1), other.data.view()).unwrap();
            self
        }
        pub fn index_mut(&mut self, index: usize) -> ArrayViewMut1<NNVFloat> {
            self.data.index_axis_mut(Axis(1), index)
        }
        #[must_use]
        pub fn get_ith_bounds(&self, index: usize) -> Self {
            Self {
                data: self
                    .data
                    .index_axis(Axis(1), index)
                    .to_owned()
                    .insert_axis(Axis(0)),
            }
        }
        #[must_use]
        pub fn unfixed_dims(&self) -> Self {
            let (lower, upper): (Vec<_>, Vec<_>) = self
                .lower()
                .iter()
                .zip(self.upper().iter())
                .filter(|(&l, &u)| l != u)
                .unzip();
            Self::new(
                Array1::from_vec(lower).view(),
                Array1::from_vec(upper).view(),
            )
        }
    }
    /// Scale by scalar
    impl<D: Dimension> Mul<NNVFloat> for Bounds<D> {
        type Output = Self;
        fn mul(self, rhs: NNVFloat) -> Self {
            Self {
                data: self.data * rhs,
            }
        }
    }
    /// Scale by scalar
    impl<D: Dimension> MulAssign<NNVFloat> for Bounds<D> {
        fn mul_assign(&mut self, rhs: NNVFloat) {
            self.data *= rhs;
        }
    }
    impl<D: Dimension + RemoveAxis> Display for Bounds<D> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::result::Result<(), std::fmt::Error> {
            f.write_fmt(::core::fmt::Arguments::new_v1(
                &["Lower: ", "\nUpper: "],
                &[
                    ::core::fmt::ArgumentV1::new_display(&self.lower()),
                    ::core::fmt::ArgumentV1::new_display(&self.upper()),
                ],
            ))
        }
    }
}
pub mod dnn {
    pub mod conv {
        #![allow(non_snake_case, clippy::module_name_repetitions)]
        //! Representation of affine transformations
        use crate::affine::Affine2;
        use crate::bounds::Bounds1;
        use crate::graph::Operation;
        use crate::star::Star2;
        use crate::tensorshape::TensorShape;
        use crate::NNVFloat;
        use itertools::Itertools;
        use ndarray::{Array1, Array2, Array3, Array4, ArrayView3};
        use serde::{Deserialize, Serialize};
        use std::any::Any;
        use std::fmt;
        use std::fmt::Debug;
        use std::ops::Deref;
        /// Assumes that data is always in a flattened state.
        /// Weights are of the shape: (`kernel_w`, `kernel_h`, `channels_in`, `channels_out`)
        pub struct Conv {
            kernel: Array4<NNVFloat>,
            bias: Array1<NNVFloat>,
            input_shape: TensorShape,
            strides: (usize, usize, usize),
            padding: ((usize, usize), (usize, usize)),
            affine: Option<Affine2>,
        }
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl ::core::clone::Clone for Conv {
            #[inline]
            fn clone(&self) -> Conv {
                match *self {
                    Conv {
                        kernel: ref __self_0_0,
                        bias: ref __self_0_1,
                        input_shape: ref __self_0_2,
                        strides: ref __self_0_3,
                        padding: ref __self_0_4,
                        affine: ref __self_0_5,
                    } => Conv {
                        kernel: ::core::clone::Clone::clone(&(*__self_0_0)),
                        bias: ::core::clone::Clone::clone(&(*__self_0_1)),
                        input_shape: ::core::clone::Clone::clone(&(*__self_0_2)),
                        strides: ::core::clone::Clone::clone(&(*__self_0_3)),
                        padding: ::core::clone::Clone::clone(&(*__self_0_4)),
                        affine: ::core::clone::Clone::clone(&(*__self_0_5)),
                    },
                }
            }
        }
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl ::core::fmt::Debug for Conv {
            fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                match *self {
                    Conv {
                        kernel: ref __self_0_0,
                        bias: ref __self_0_1,
                        input_shape: ref __self_0_2,
                        strides: ref __self_0_3,
                        padding: ref __self_0_4,
                        affine: ref __self_0_5,
                    } => {
                        let debug_trait_builder =
                            &mut ::core::fmt::Formatter::debug_struct(f, "Conv");
                        let _ = ::core::fmt::DebugStruct::field(
                            debug_trait_builder,
                            "kernel",
                            &&(*__self_0_0),
                        );
                        let _ = ::core::fmt::DebugStruct::field(
                            debug_trait_builder,
                            "bias",
                            &&(*__self_0_1),
                        );
                        let _ = ::core::fmt::DebugStruct::field(
                            debug_trait_builder,
                            "input_shape",
                            &&(*__self_0_2),
                        );
                        let _ = ::core::fmt::DebugStruct::field(
                            debug_trait_builder,
                            "strides",
                            &&(*__self_0_3),
                        );
                        let _ = ::core::fmt::DebugStruct::field(
                            debug_trait_builder,
                            "padding",
                            &&(*__self_0_4),
                        );
                        let _ = ::core::fmt::DebugStruct::field(
                            debug_trait_builder,
                            "affine",
                            &&(*__self_0_5),
                        );
                        ::core::fmt::DebugStruct::finish(debug_trait_builder)
                    }
                }
            }
        }
        #[doc(hidden)]
        #[allow(non_upper_case_globals, unused_attributes, unused_qualifications)]
        const _: () = {
            #[allow(unused_extern_crates, clippy::useless_attribute)]
            extern crate serde as _serde;
            #[automatically_derived]
            impl<'de> _serde::Deserialize<'de> for Conv {
                fn deserialize<__D>(
                    __deserializer: __D,
                ) -> _serde::__private::Result<Self, __D::Error>
                where
                    __D: _serde::Deserializer<'de>,
                {
                    #[allow(non_camel_case_types)]
                    enum __Field {
                        __field0,
                        __field1,
                        __field2,
                        __field3,
                        __field4,
                        __field5,
                        __ignore,
                    }
                    struct __FieldVisitor;
                    impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                        type Value = __Field;
                        fn expecting(
                            &self,
                            __formatter: &mut _serde::__private::Formatter,
                        ) -> _serde::__private::fmt::Result {
                            _serde::__private::Formatter::write_str(__formatter, "field identifier")
                        }
                        fn visit_u64<__E>(
                            self,
                            __value: u64,
                        ) -> _serde::__private::Result<Self::Value, __E>
                        where
                            __E: _serde::de::Error,
                        {
                            match __value {
                                0u64 => _serde::__private::Ok(__Field::__field0),
                                1u64 => _serde::__private::Ok(__Field::__field1),
                                2u64 => _serde::__private::Ok(__Field::__field2),
                                3u64 => _serde::__private::Ok(__Field::__field3),
                                4u64 => _serde::__private::Ok(__Field::__field4),
                                5u64 => _serde::__private::Ok(__Field::__field5),
                                _ => _serde::__private::Ok(__Field::__ignore),
                            }
                        }
                        fn visit_str<__E>(
                            self,
                            __value: &str,
                        ) -> _serde::__private::Result<Self::Value, __E>
                        where
                            __E: _serde::de::Error,
                        {
                            match __value {
                                "kernel" => _serde::__private::Ok(__Field::__field0),
                                "bias" => _serde::__private::Ok(__Field::__field1),
                                "input_shape" => _serde::__private::Ok(__Field::__field2),
                                "strides" => _serde::__private::Ok(__Field::__field3),
                                "padding" => _serde::__private::Ok(__Field::__field4),
                                "affine" => _serde::__private::Ok(__Field::__field5),
                                _ => _serde::__private::Ok(__Field::__ignore),
                            }
                        }
                        fn visit_bytes<__E>(
                            self,
                            __value: &[u8],
                        ) -> _serde::__private::Result<Self::Value, __E>
                        where
                            __E: _serde::de::Error,
                        {
                            match __value {
                                b"kernel" => _serde::__private::Ok(__Field::__field0),
                                b"bias" => _serde::__private::Ok(__Field::__field1),
                                b"input_shape" => _serde::__private::Ok(__Field::__field2),
                                b"strides" => _serde::__private::Ok(__Field::__field3),
                                b"padding" => _serde::__private::Ok(__Field::__field4),
                                b"affine" => _serde::__private::Ok(__Field::__field5),
                                _ => _serde::__private::Ok(__Field::__ignore),
                            }
                        }
                    }
                    impl<'de> _serde::Deserialize<'de> for __Field {
                        #[inline]
                        fn deserialize<__D>(
                            __deserializer: __D,
                        ) -> _serde::__private::Result<Self, __D::Error>
                        where
                            __D: _serde::Deserializer<'de>,
                        {
                            _serde::Deserializer::deserialize_identifier(
                                __deserializer,
                                __FieldVisitor,
                            )
                        }
                    }
                    struct __Visitor<'de> {
                        marker: _serde::__private::PhantomData<Conv>,
                        lifetime: _serde::__private::PhantomData<&'de ()>,
                    }
                    impl<'de> _serde::de::Visitor<'de> for __Visitor<'de> {
                        type Value = Conv;
                        fn expecting(
                            &self,
                            __formatter: &mut _serde::__private::Formatter,
                        ) -> _serde::__private::fmt::Result {
                            _serde::__private::Formatter::write_str(__formatter, "struct Conv")
                        }
                        #[inline]
                        fn visit_seq<__A>(
                            self,
                            mut __seq: __A,
                        ) -> _serde::__private::Result<Self::Value, __A::Error>
                        where
                            __A: _serde::de::SeqAccess<'de>,
                        {
                            let __field0 = match match _serde::de::SeqAccess::next_element::<
                                Array4<NNVFloat>,
                            >(&mut __seq)
                            {
                                _serde::__private::Ok(__val) => __val,
                                _serde::__private::Err(__err) => {
                                    return _serde::__private::Err(__err);
                                }
                            } {
                                _serde::__private::Some(__value) => __value,
                                _serde::__private::None => {
                                    return _serde::__private::Err(
                                        _serde::de::Error::invalid_length(
                                            0usize,
                                            &"struct Conv with 6 elements",
                                        ),
                                    );
                                }
                            };
                            let __field1 = match match _serde::de::SeqAccess::next_element::<
                                Array1<NNVFloat>,
                            >(&mut __seq)
                            {
                                _serde::__private::Ok(__val) => __val,
                                _serde::__private::Err(__err) => {
                                    return _serde::__private::Err(__err);
                                }
                            } {
                                _serde::__private::Some(__value) => __value,
                                _serde::__private::None => {
                                    return _serde::__private::Err(
                                        _serde::de::Error::invalid_length(
                                            1usize,
                                            &"struct Conv with 6 elements",
                                        ),
                                    );
                                }
                            };
                            let __field2 = match match _serde::de::SeqAccess::next_element::<
                                TensorShape,
                            >(&mut __seq)
                            {
                                _serde::__private::Ok(__val) => __val,
                                _serde::__private::Err(__err) => {
                                    return _serde::__private::Err(__err);
                                }
                            } {
                                _serde::__private::Some(__value) => __value,
                                _serde::__private::None => {
                                    return _serde::__private::Err(
                                        _serde::de::Error::invalid_length(
                                            2usize,
                                            &"struct Conv with 6 elements",
                                        ),
                                    );
                                }
                            };
                            let __field3 = match match _serde::de::SeqAccess::next_element::<(
                                usize,
                                usize,
                                usize,
                            )>(&mut __seq)
                            {
                                _serde::__private::Ok(__val) => __val,
                                _serde::__private::Err(__err) => {
                                    return _serde::__private::Err(__err);
                                }
                            } {
                                _serde::__private::Some(__value) => __value,
                                _serde::__private::None => {
                                    return _serde::__private::Err(
                                        _serde::de::Error::invalid_length(
                                            3usize,
                                            &"struct Conv with 6 elements",
                                        ),
                                    );
                                }
                            };
                            let __field4 = match match _serde::de::SeqAccess::next_element::<(
                                (usize, usize),
                                (usize, usize),
                            )>(&mut __seq)
                            {
                                _serde::__private::Ok(__val) => __val,
                                _serde::__private::Err(__err) => {
                                    return _serde::__private::Err(__err);
                                }
                            } {
                                _serde::__private::Some(__value) => __value,
                                _serde::__private::None => {
                                    return _serde::__private::Err(
                                        _serde::de::Error::invalid_length(
                                            4usize,
                                            &"struct Conv with 6 elements",
                                        ),
                                    );
                                }
                            };
                            let __field5 = match match _serde::de::SeqAccess::next_element::<
                                Option<Affine2>,
                            >(&mut __seq)
                            {
                                _serde::__private::Ok(__val) => __val,
                                _serde::__private::Err(__err) => {
                                    return _serde::__private::Err(__err);
                                }
                            } {
                                _serde::__private::Some(__value) => __value,
                                _serde::__private::None => {
                                    return _serde::__private::Err(
                                        _serde::de::Error::invalid_length(
                                            5usize,
                                            &"struct Conv with 6 elements",
                                        ),
                                    );
                                }
                            };
                            _serde::__private::Ok(Conv {
                                kernel: __field0,
                                bias: __field1,
                                input_shape: __field2,
                                strides: __field3,
                                padding: __field4,
                                affine: __field5,
                            })
                        }
                        #[inline]
                        fn visit_map<__A>(
                            self,
                            mut __map: __A,
                        ) -> _serde::__private::Result<Self::Value, __A::Error>
                        where
                            __A: _serde::de::MapAccess<'de>,
                        {
                            let mut __field0: _serde::__private::Option<Array4<NNVFloat>> =
                                _serde::__private::None;
                            let mut __field1: _serde::__private::Option<Array1<NNVFloat>> =
                                _serde::__private::None;
                            let mut __field2: _serde::__private::Option<TensorShape> =
                                _serde::__private::None;
                            let mut __field3: _serde::__private::Option<(usize, usize, usize)> =
                                _serde::__private::None;
                            let mut __field4: _serde::__private::Option<(
                                (usize, usize),
                                (usize, usize),
                            )> = _serde::__private::None;
                            let mut __field5: _serde::__private::Option<Option<Affine2>> =
                                _serde::__private::None;
                            while let _serde::__private::Some(__key) =
                                match _serde::de::MapAccess::next_key::<__Field>(&mut __map) {
                                    _serde::__private::Ok(__val) => __val,
                                    _serde::__private::Err(__err) => {
                                        return _serde::__private::Err(__err);
                                    }
                                }
                            {
                                match __key {
                                    __Field::__field0 => {
                                        if _serde::__private::Option::is_some(&__field0) {
                                            return _serde::__private::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field(
                                                    "kernel",
                                                ),
                                            );
                                        }
                                        __field0 = _serde::__private::Some(
                                            match _serde::de::MapAccess::next_value::<
                                                Array4<NNVFloat>,
                                            >(
                                                &mut __map
                                            ) {
                                                _serde::__private::Ok(__val) => __val,
                                                _serde::__private::Err(__err) => {
                                                    return _serde::__private::Err(__err);
                                                }
                                            },
                                        );
                                    }
                                    __Field::__field1 => {
                                        if _serde::__private::Option::is_some(&__field1) {
                                            return _serde::__private::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field(
                                                    "bias",
                                                ),
                                            );
                                        }
                                        __field1 = _serde::__private::Some(
                                            match _serde::de::MapAccess::next_value::<
                                                Array1<NNVFloat>,
                                            >(
                                                &mut __map
                                            ) {
                                                _serde::__private::Ok(__val) => __val,
                                                _serde::__private::Err(__err) => {
                                                    return _serde::__private::Err(__err);
                                                }
                                            },
                                        );
                                    }
                                    __Field::__field2 => {
                                        if _serde::__private::Option::is_some(&__field2) {
                                            return _serde::__private::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field(
                                                    "input_shape",
                                                ),
                                            );
                                        }
                                        __field2 = _serde::__private::Some(
                                            match _serde::de::MapAccess::next_value::<TensorShape>(
                                                &mut __map,
                                            ) {
                                                _serde::__private::Ok(__val) => __val,
                                                _serde::__private::Err(__err) => {
                                                    return _serde::__private::Err(__err);
                                                }
                                            },
                                        );
                                    }
                                    __Field::__field3 => {
                                        if _serde::__private::Option::is_some(&__field3) {
                                            return _serde::__private::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field(
                                                    "strides",
                                                ),
                                            );
                                        }
                                        __field3 = _serde::__private::Some(
                                            match _serde::de::MapAccess::next_value::<(
                                                usize,
                                                usize,
                                                usize,
                                            )>(
                                                &mut __map
                                            ) {
                                                _serde::__private::Ok(__val) => __val,
                                                _serde::__private::Err(__err) => {
                                                    return _serde::__private::Err(__err);
                                                }
                                            },
                                        );
                                    }
                                    __Field::__field4 => {
                                        if _serde::__private::Option::is_some(&__field4) {
                                            return _serde::__private::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field(
                                                    "padding",
                                                ),
                                            );
                                        }
                                        __field4 = _serde::__private::Some(
                                            match _serde::de::MapAccess::next_value::<(
                                                (usize, usize),
                                                (usize, usize),
                                            )>(
                                                &mut __map
                                            ) {
                                                _serde::__private::Ok(__val) => __val,
                                                _serde::__private::Err(__err) => {
                                                    return _serde::__private::Err(__err);
                                                }
                                            },
                                        );
                                    }
                                    __Field::__field5 => {
                                        if _serde::__private::Option::is_some(&__field5) {
                                            return _serde::__private::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field(
                                                    "affine",
                                                ),
                                            );
                                        }
                                        __field5 = _serde::__private::Some(
                                            match _serde::de::MapAccess::next_value::<Option<Affine2>>(
                                                &mut __map,
                                            ) {
                                                _serde::__private::Ok(__val) => __val,
                                                _serde::__private::Err(__err) => {
                                                    return _serde::__private::Err(__err);
                                                }
                                            },
                                        );
                                    }
                                    _ => {
                                        let _ = match _serde::de::MapAccess::next_value::<
                                            _serde::de::IgnoredAny,
                                        >(
                                            &mut __map
                                        ) {
                                            _serde::__private::Ok(__val) => __val,
                                            _serde::__private::Err(__err) => {
                                                return _serde::__private::Err(__err);
                                            }
                                        };
                                    }
                                }
                            }
                            let __field0 = match __field0 {
                                _serde::__private::Some(__field0) => __field0,
                                _serde::__private::None => {
                                    match _serde::__private::de::missing_field("kernel") {
                                        _serde::__private::Ok(__val) => __val,
                                        _serde::__private::Err(__err) => {
                                            return _serde::__private::Err(__err);
                                        }
                                    }
                                }
                            };
                            let __field1 = match __field1 {
                                _serde::__private::Some(__field1) => __field1,
                                _serde::__private::None => {
                                    match _serde::__private::de::missing_field("bias") {
                                        _serde::__private::Ok(__val) => __val,
                                        _serde::__private::Err(__err) => {
                                            return _serde::__private::Err(__err);
                                        }
                                    }
                                }
                            };
                            let __field2 = match __field2 {
                                _serde::__private::Some(__field2) => __field2,
                                _serde::__private::None => {
                                    match _serde::__private::de::missing_field("input_shape") {
                                        _serde::__private::Ok(__val) => __val,
                                        _serde::__private::Err(__err) => {
                                            return _serde::__private::Err(__err);
                                        }
                                    }
                                }
                            };
                            let __field3 = match __field3 {
                                _serde::__private::Some(__field3) => __field3,
                                _serde::__private::None => {
                                    match _serde::__private::de::missing_field("strides") {
                                        _serde::__private::Ok(__val) => __val,
                                        _serde::__private::Err(__err) => {
                                            return _serde::__private::Err(__err);
                                        }
                                    }
                                }
                            };
                            let __field4 = match __field4 {
                                _serde::__private::Some(__field4) => __field4,
                                _serde::__private::None => {
                                    match _serde::__private::de::missing_field("padding") {
                                        _serde::__private::Ok(__val) => __val,
                                        _serde::__private::Err(__err) => {
                                            return _serde::__private::Err(__err);
                                        }
                                    }
                                }
                            };
                            let __field5 = match __field5 {
                                _serde::__private::Some(__field5) => __field5,
                                _serde::__private::None => {
                                    match _serde::__private::de::missing_field("affine") {
                                        _serde::__private::Ok(__val) => __val,
                                        _serde::__private::Err(__err) => {
                                            return _serde::__private::Err(__err);
                                        }
                                    }
                                }
                            };
                            _serde::__private::Ok(Conv {
                                kernel: __field0,
                                bias: __field1,
                                input_shape: __field2,
                                strides: __field3,
                                padding: __field4,
                                affine: __field5,
                            })
                        }
                    }
                    const FIELDS: &'static [&'static str] = &[
                        "kernel",
                        "bias",
                        "input_shape",
                        "strides",
                        "padding",
                        "affine",
                    ];
                    _serde::Deserializer::deserialize_struct(
                        __deserializer,
                        "Conv",
                        FIELDS,
                        __Visitor {
                            marker: _serde::__private::PhantomData::<Conv>,
                            lifetime: _serde::__private::PhantomData,
                        },
                    )
                }
            }
        };
        #[doc(hidden)]
        #[allow(non_upper_case_globals, unused_attributes, unused_qualifications)]
        const _: () = {
            #[allow(unused_extern_crates, clippy::useless_attribute)]
            extern crate serde as _serde;
            #[automatically_derived]
            impl _serde::Serialize for Conv {
                fn serialize<__S>(
                    &self,
                    __serializer: __S,
                ) -> _serde::__private::Result<__S::Ok, __S::Error>
                where
                    __S: _serde::Serializer,
                {
                    let mut __serde_state = match _serde::Serializer::serialize_struct(
                        __serializer,
                        "Conv",
                        false as usize + 1 + 1 + 1 + 1 + 1 + 1,
                    ) {
                        _serde::__private::Ok(__val) => __val,
                        _serde::__private::Err(__err) => {
                            return _serde::__private::Err(__err);
                        }
                    };
                    match _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "kernel",
                        &self.kernel,
                    ) {
                        _serde::__private::Ok(__val) => __val,
                        _serde::__private::Err(__err) => {
                            return _serde::__private::Err(__err);
                        }
                    };
                    match _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "bias",
                        &self.bias,
                    ) {
                        _serde::__private::Ok(__val) => __val,
                        _serde::__private::Err(__err) => {
                            return _serde::__private::Err(__err);
                        }
                    };
                    match _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "input_shape",
                        &self.input_shape,
                    ) {
                        _serde::__private::Ok(__val) => __val,
                        _serde::__private::Err(__err) => {
                            return _serde::__private::Err(__err);
                        }
                    };
                    match _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "strides",
                        &self.strides,
                    ) {
                        _serde::__private::Ok(__val) => __val,
                        _serde::__private::Err(__err) => {
                            return _serde::__private::Err(__err);
                        }
                    };
                    match _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "padding",
                        &self.padding,
                    ) {
                        _serde::__private::Ok(__val) => __val,
                        _serde::__private::Err(__err) => {
                            return _serde::__private::Err(__err);
                        }
                    };
                    match _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "affine",
                        &self.affine,
                    ) {
                        _serde::__private::Ok(__val) => __val,
                        _serde::__private::Err(__err) => {
                            return _serde::__private::Err(__err);
                        }
                    };
                    _serde::ser::SerializeStruct::end(__serde_state)
                }
            }
        };
        impl Conv {
            /// # Panics
            /// If improper shapes are passed in
            pub fn new(
                kernel: Array4<NNVFloat>,
                bias: Array1<NNVFloat>,
                input_shape: TensorShape,
                strides: (usize, usize, usize),
                padding: ((usize, usize), (usize, usize)),
            ) -> Self {
                if true {
                    match (&kernel.shape()[3], &bias.len()) {
                        (left_val, right_val) => {
                            if !(*left_val == *right_val) {
                                let kind = ::core::panicking::AssertKind::Eq;
                                ::core::panicking::assert_failed(
                                    kind,
                                    &*left_val,
                                    &*right_val,
                                    ::core::option::Option::None,
                                );
                            }
                        }
                    };
                };
                if strides.2 != 1 {
                    ::core::panicking::panic("not yet implemented");
                }
                let mut s = Self {
                    kernel,
                    bias,
                    input_shape,
                    strides,
                    padding,
                    affine: None,
                };
                s.construct_affine();
                s
            }
            /// # Panics
            pub fn get_affine(&self) -> &Affine2 {
                self.affine.as_ref().unwrap()
            }
            pub fn input_shape(&self) -> TensorShape {
                self.input_shape.clone()
            }
            /// # Panics
            pub fn output_shape(&self) -> TensorShape {
                let k_h = self.kernel.shape()[0];
                let k_w = self.kernel.shape()[1];
                let h_out = (self.input_shape[1].unwrap() + self.padding.0 .0 + self.padding.0 .1
                    - (k_h - 1)
                    - 1)
                    / self.strides.0
                    + 1;
                let w_out = (self.input_shape[2].unwrap() + self.padding.1 .0 + self.padding.1 .1
                    - (k_w - 1)
                    - 1)
                    / self.strides.1
                    + 1;
                TensorShape::new(<[_]>::into_vec(box [
                    None,
                    Some(h_out),
                    Some(w_out),
                    Some(self.kernel.shape()[3]),
                ]))
            }
            /// # Panics
            fn construct_affine(&mut self) {
                let h_in = self.input_shape[1].unwrap();
                let w_in = self.input_shape[2].unwrap();
                let c_in = self.input_shape[3].unwrap();
                let h_out = self.output_shape()[1].unwrap();
                let w_out = self.output_shape()[2].unwrap();
                let c_out = self.output_shape()[3].unwrap();
                let k_h = self.kernel.shape()[0];
                let k_w = self.kernel.shape()[1];
                let input_dims = h_in * w_in * c_in;
                let output_dims = h_out * w_out * c_out;
                let mut weight = Array2::<NNVFloat>::zeros((output_dims, input_dims));
                for (y_out, x_out) in (0..h_out).cartesian_product(0..w_out) {
                    let y_0 = y_out * self.strides.0;
                    let x_0 = x_out * self.strides.1;
                    for k_y in 0..k_h {
                        if y_0 + k_y < self.padding.0 .0 || y_0 + k_y >= h_in + self.padding.0 .0 {
                            continue;
                        }
                        let y_in = y_0 + k_y - self.padding.0 .0;
                        for k_x in 0..k_w {
                            if x_0 + k_x < self.padding.1 .0
                                || x_0 + k_x >= w_in + self.padding.1 .0
                            {
                                continue;
                            }
                            let x_in = x_0 + k_x - self.padding.1 .0;
                            for f_in in 0..c_in {
                                let input_idx = y_in * (w_in * c_in) + x_in * c_in + f_in;
                                for f_out in 0..c_out {
                                    let output_idx =
                                        y_out * (w_out * c_out) + x_out * c_out + f_out;
                                    weight[[output_idx, input_idx]] =
                                        self.kernel[[k_y, k_x, f_in, f_out]];
                                }
                            }
                        }
                    }
                }
                let bias = (Array3::<NNVFloat>::ones((h_out, w_out, c_out))
                    * self.bias.view().into_shape((1, 1, c_out)).unwrap())
                .into_shape(h_out * w_out * c_out)
                .unwrap();
                self.affine = Some(Affine2::new(weight, bias));
            }
            /// # Panics
            pub fn convolve(&self, data: ArrayView3<NNVFloat>) -> Array3<NNVFloat> {
                let h_in = self.input_shape[1].unwrap();
                let w_in = self.input_shape[2].unwrap();
                let c_in = self.input_shape[3].unwrap();
                let h_out = self.output_shape()[1].unwrap();
                let w_out = self.output_shape()[2].unwrap();
                let c_out = self.output_shape()[3].unwrap();
                let k_h = self.kernel.shape()[0];
                let k_w = self.kernel.shape()[1];
                let input_shape = <[_]>::into_vec(box [h_in, w_in, c_in]);
                let output_shape = (h_out, w_out, c_out);
                match (&data.shape(), &input_shape) {
                    (left_val, right_val) => {
                        if !(*left_val == *right_val) {
                            let kind = ::core::panicking::AssertKind::Eq;
                            ::core::panicking::assert_failed(
                                kind,
                                &*left_val,
                                &*right_val,
                                ::core::option::Option::None,
                            );
                        }
                    }
                };
                let mut output = Array3::<NNVFloat>::ones(output_shape);
                output = output * self.bias.view().into_shape((1, 1, c_out)).unwrap();
                for (y_out, x_out) in (0..h_out).cartesian_product(0..w_out) {
                    let y_0 = y_out * self.strides.0;
                    let x_0 = x_out * self.strides.1;
                    for k_y in 0..k_h {
                        if y_0 + k_y < self.padding.0 .0 || y_0 + k_y >= h_in + self.padding.0 .0 {
                            continue;
                        }
                        let y_in = y_0 + k_y - self.padding.0 .0;
                        for k_x in 0..k_w {
                            if x_0 + k_x < self.padding.1 .0
                                || x_0 + k_x >= w_in + self.padding.1 .0
                            {
                                continue;
                            }
                            let x_in = x_0 + k_x - self.padding.1 .0;
                            for f_in in 0..c_in {
                                for f_out in 0..c_out {
                                    output[[y_out, x_out, f_out]] += data[[y_in, x_in, f_in]]
                                        * self.kernel[[k_y, k_x, f_in, f_out]];
                                }
                            }
                        }
                    }
                }
                output
            }
        }
        impl Operation for Conv {
            fn input_shapes(&self) -> Vec<TensorShape> {
                <[_]>::into_vec(box [TensorShape::new(<[_]>::into_vec(box [Some(
                    self.get_affine().input_dim(),
                )]))])
            }
            fn output_shapes(&self) -> Vec<TensorShape> {
                <[_]>::into_vec(box [TensorShape::new(<[_]>::into_vec(box [Some(
                    self.get_affine().output_dim(),
                )]))])
            }
            fn forward1(&self, input: &[&Array1<NNVFloat>]) -> Vec<Array1<NNVFloat>> {
                match (&input.len(), &1) {
                    (left_val, right_val) => {
                        if !(*left_val == *right_val) {
                            let kind = ::core::panicking::AssertKind::Eq;
                            ::core::panicking::assert_failed(
                                kind,
                                &*left_val,
                                &*right_val,
                                ::core::option::Option::None,
                            );
                        }
                    }
                };
                let input = input.first().unwrap();
                if true {
                    match (&input.ndim(), &1) {
                        (left_val, right_val) => {
                            if !(*left_val == *right_val) {
                                let kind = ::core::panicking::AssertKind::Eq;
                                ::core::panicking::assert_failed(
                                    kind,
                                    &*left_val,
                                    &*right_val,
                                    ::core::option::Option::None,
                                );
                            }
                        }
                    };
                };
                <[_]>::into_vec(box [self.get_affine().apply(&input.view())])
            }
            fn forward2(&self, input: &[&Array2<NNVFloat>]) -> Vec<Array2<NNVFloat>> {
                match (&input.len(), &1) {
                    (left_val, right_val) => {
                        if !(*left_val == *right_val) {
                            let kind = ::core::panicking::AssertKind::Eq;
                            ::core::panicking::assert_failed(
                                kind,
                                &*left_val,
                                &*right_val,
                                ::core::option::Option::None,
                            );
                        }
                    }
                };
                let input = input.first().unwrap();
                <[_]>::into_vec(box [self.get_affine().apply_matrix(&input.view())])
            }
            fn apply_bounds(
                &self,
                bounds: &[Bounds1],
                lower_aff: &[Affine2],
                upper_aff: &[Affine2],
            ) -> Vec<(Bounds1, Affine2, Affine2)> {
                match (&bounds.len(), &1) {
                    (left_val, right_val) => {
                        if !(*left_val == *right_val) {
                            let kind = ::core::panicking::AssertKind::Eq;
                            ::core::panicking::assert_failed(
                                kind,
                                &*left_val,
                                &*right_val,
                                ::core::option::Option::None,
                            );
                        }
                    }
                };
                match (&lower_aff.len(), &1) {
                    (left_val, right_val) => {
                        if !(*left_val == *right_val) {
                            let kind = ::core::panicking::AssertKind::Eq;
                            ::core::panicking::assert_failed(
                                kind,
                                &*left_val,
                                &*right_val,
                                ::core::option::Option::None,
                            );
                        }
                    }
                };
                match (&upper_aff.len(), &1) {
                    (left_val, right_val) => {
                        if !(*left_val == *right_val) {
                            let kind = ::core::panicking::AssertKind::Eq;
                            ::core::panicking::assert_failed(
                                kind,
                                &*left_val,
                                &*right_val,
                                ::core::option::Option::None,
                            );
                        }
                    }
                };
                let bounds = bounds.first().unwrap();
                let lower_aff = lower_aff.first().unwrap();
                let upper_aff = upper_aff.first().unwrap();
                let new_lower = self.get_affine().signed_compose(lower_aff, upper_aff);
                let new_upper = self.get_affine().signed_compose(upper_aff, lower_aff);
                <[_]>::into_vec(box [(
                    self.get_affine().signed_apply(bounds),
                    new_lower,
                    new_upper,
                )])
            }
            fn as_any(&self) -> &dyn Any {
                self
            }
            fn forward_star<StarRef: Deref<Target = Star2>>(
                &self,
                stars: Vec<StarRef>,
                _activation_idx: Option<usize>,
                parent_axis_aligned_input_bounds: Vec<&Bounds1>,
            ) -> (Vec<Star2>, Vec<Bounds1>, bool) {
                match (&1, &stars.len()) {
                    (left_val, right_val) => {
                        if !(*left_val == *right_val) {
                            let kind = ::core::panicking::AssertKind::Eq;
                            ::core::panicking::assert_failed(
                                kind,
                                &*left_val,
                                &*right_val,
                                ::core::option::Option::None,
                            );
                        }
                    }
                };
                match (&1, &parent_axis_aligned_input_bounds.len()) {
                    (left_val, right_val) => {
                        if !(*left_val == *right_val) {
                            let kind = ::core::panicking::AssertKind::Eq;
                            ::core::panicking::assert_failed(
                                kind,
                                &*left_val,
                                &*right_val,
                                ::core::option::Option::None,
                            );
                        }
                    }
                };
                (
                    <[_]>::into_vec(box [stars[0].affine_map2(self.get_affine())]),
                    <[_]>::into_vec(box [parent_axis_aligned_input_bounds[0].clone()]),
                    false,
                )
            }
        }
        impl fmt::Display for Conv {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_fmt(::core::fmt::Arguments::new_v1(
                    &["Conv ", "x", ", "],
                    &[
                        ::core::fmt::ArgumentV1::new_display(&self.kernel.shape()[1]),
                        ::core::fmt::ArgumentV1::new_display(&self.kernel.shape()[0]),
                        ::core::fmt::ArgumentV1::new_display(&self.kernel.shape()[2]),
                    ],
                ))
            }
        }
    }
    pub mod dense {
        use crate::affine::Affine2;
        use crate::bounds::Bounds1;
        use crate::graph::Operation;
        use crate::star::Star2;
        use crate::tensorshape::TensorShape;
        use crate::NNVFloat;
        use ndarray::Array1;
        use ndarray::Array2;
        use serde::{Deserialize, Serialize};
        use std::fmt;
        use std::ops::Deref;
        pub struct Dense {
            aff: Affine2,
        }
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl ::core::clone::Clone for Dense {
            #[inline]
            fn clone(&self) -> Dense {
                match *self {
                    Dense {
                        aff: ref __self_0_0,
                    } => Dense {
                        aff: ::core::clone::Clone::clone(&(*__self_0_0)),
                    },
                }
            }
        }
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl ::core::fmt::Debug for Dense {
            fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                match *self {
                    Dense {
                        aff: ref __self_0_0,
                    } => {
                        let debug_trait_builder =
                            &mut ::core::fmt::Formatter::debug_struct(f, "Dense");
                        let _ = ::core::fmt::DebugStruct::field(
                            debug_trait_builder,
                            "aff",
                            &&(*__self_0_0),
                        );
                        ::core::fmt::DebugStruct::finish(debug_trait_builder)
                    }
                }
            }
        }
        #[doc(hidden)]
        #[allow(non_upper_case_globals, unused_attributes, unused_qualifications)]
        const _: () = {
            #[allow(unused_extern_crates, clippy::useless_attribute)]
            extern crate serde as _serde;
            #[automatically_derived]
            impl<'de> _serde::Deserialize<'de> for Dense {
                fn deserialize<__D>(
                    __deserializer: __D,
                ) -> _serde::__private::Result<Self, __D::Error>
                where
                    __D: _serde::Deserializer<'de>,
                {
                    #[allow(non_camel_case_types)]
                    enum __Field {
                        __field0,
                        __ignore,
                    }
                    struct __FieldVisitor;
                    impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                        type Value = __Field;
                        fn expecting(
                            &self,
                            __formatter: &mut _serde::__private::Formatter,
                        ) -> _serde::__private::fmt::Result {
                            _serde::__private::Formatter::write_str(__formatter, "field identifier")
                        }
                        fn visit_u64<__E>(
                            self,
                            __value: u64,
                        ) -> _serde::__private::Result<Self::Value, __E>
                        where
                            __E: _serde::de::Error,
                        {
                            match __value {
                                0u64 => _serde::__private::Ok(__Field::__field0),
                                _ => _serde::__private::Ok(__Field::__ignore),
                            }
                        }
                        fn visit_str<__E>(
                            self,
                            __value: &str,
                        ) -> _serde::__private::Result<Self::Value, __E>
                        where
                            __E: _serde::de::Error,
                        {
                            match __value {
                                "aff" => _serde::__private::Ok(__Field::__field0),
                                _ => _serde::__private::Ok(__Field::__ignore),
                            }
                        }
                        fn visit_bytes<__E>(
                            self,
                            __value: &[u8],
                        ) -> _serde::__private::Result<Self::Value, __E>
                        where
                            __E: _serde::de::Error,
                        {
                            match __value {
                                b"aff" => _serde::__private::Ok(__Field::__field0),
                                _ => _serde::__private::Ok(__Field::__ignore),
                            }
                        }
                    }
                    impl<'de> _serde::Deserialize<'de> for __Field {
                        #[inline]
                        fn deserialize<__D>(
                            __deserializer: __D,
                        ) -> _serde::__private::Result<Self, __D::Error>
                        where
                            __D: _serde::Deserializer<'de>,
                        {
                            _serde::Deserializer::deserialize_identifier(
                                __deserializer,
                                __FieldVisitor,
                            )
                        }
                    }
                    struct __Visitor<'de> {
                        marker: _serde::__private::PhantomData<Dense>,
                        lifetime: _serde::__private::PhantomData<&'de ()>,
                    }
                    impl<'de> _serde::de::Visitor<'de> for __Visitor<'de> {
                        type Value = Dense;
                        fn expecting(
                            &self,
                            __formatter: &mut _serde::__private::Formatter,
                        ) -> _serde::__private::fmt::Result {
                            _serde::__private::Formatter::write_str(__formatter, "struct Dense")
                        }
                        #[inline]
                        fn visit_seq<__A>(
                            self,
                            mut __seq: __A,
                        ) -> _serde::__private::Result<Self::Value, __A::Error>
                        where
                            __A: _serde::de::SeqAccess<'de>,
                        {
                            let __field0 = match match _serde::de::SeqAccess::next_element::<Affine2>(
                                &mut __seq,
                            ) {
                                _serde::__private::Ok(__val) => __val,
                                _serde::__private::Err(__err) => {
                                    return _serde::__private::Err(__err);
                                }
                            } {
                                _serde::__private::Some(__value) => __value,
                                _serde::__private::None => {
                                    return _serde::__private::Err(
                                        _serde::de::Error::invalid_length(
                                            0usize,
                                            &"struct Dense with 1 element",
                                        ),
                                    );
                                }
                            };
                            _serde::__private::Ok(Dense { aff: __field0 })
                        }
                        #[inline]
                        fn visit_map<__A>(
                            self,
                            mut __map: __A,
                        ) -> _serde::__private::Result<Self::Value, __A::Error>
                        where
                            __A: _serde::de::MapAccess<'de>,
                        {
                            let mut __field0: _serde::__private::Option<Affine2> =
                                _serde::__private::None;
                            while let _serde::__private::Some(__key) =
                                match _serde::de::MapAccess::next_key::<__Field>(&mut __map) {
                                    _serde::__private::Ok(__val) => __val,
                                    _serde::__private::Err(__err) => {
                                        return _serde::__private::Err(__err);
                                    }
                                }
                            {
                                match __key {
                                    __Field::__field0 => {
                                        if _serde::__private::Option::is_some(&__field0) {
                                            return _serde::__private::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field(
                                                    "aff",
                                                ),
                                            );
                                        }
                                        __field0 = _serde::__private::Some(
                                            match _serde::de::MapAccess::next_value::<Affine2>(
                                                &mut __map,
                                            ) {
                                                _serde::__private::Ok(__val) => __val,
                                                _serde::__private::Err(__err) => {
                                                    return _serde::__private::Err(__err);
                                                }
                                            },
                                        );
                                    }
                                    _ => {
                                        let _ = match _serde::de::MapAccess::next_value::<
                                            _serde::de::IgnoredAny,
                                        >(
                                            &mut __map
                                        ) {
                                            _serde::__private::Ok(__val) => __val,
                                            _serde::__private::Err(__err) => {
                                                return _serde::__private::Err(__err);
                                            }
                                        };
                                    }
                                }
                            }
                            let __field0 = match __field0 {
                                _serde::__private::Some(__field0) => __field0,
                                _serde::__private::None => {
                                    match _serde::__private::de::missing_field("aff") {
                                        _serde::__private::Ok(__val) => __val,
                                        _serde::__private::Err(__err) => {
                                            return _serde::__private::Err(__err);
                                        }
                                    }
                                }
                            };
                            _serde::__private::Ok(Dense { aff: __field0 })
                        }
                    }
                    const FIELDS: &'static [&'static str] = &["aff"];
                    _serde::Deserializer::deserialize_struct(
                        __deserializer,
                        "Dense",
                        FIELDS,
                        __Visitor {
                            marker: _serde::__private::PhantomData::<Dense>,
                            lifetime: _serde::__private::PhantomData,
                        },
                    )
                }
            }
        };
        #[doc(hidden)]
        #[allow(non_upper_case_globals, unused_attributes, unused_qualifications)]
        const _: () = {
            #[allow(unused_extern_crates, clippy::useless_attribute)]
            extern crate serde as _serde;
            #[automatically_derived]
            impl _serde::Serialize for Dense {
                fn serialize<__S>(
                    &self,
                    __serializer: __S,
                ) -> _serde::__private::Result<__S::Ok, __S::Error>
                where
                    __S: _serde::Serializer,
                {
                    let mut __serde_state = match _serde::Serializer::serialize_struct(
                        __serializer,
                        "Dense",
                        false as usize + 1,
                    ) {
                        _serde::__private::Ok(__val) => __val,
                        _serde::__private::Err(__err) => {
                            return _serde::__private::Err(__err);
                        }
                    };
                    match _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "aff",
                        &self.aff,
                    ) {
                        _serde::__private::Ok(__val) => __val,
                        _serde::__private::Err(__err) => {
                            return _serde::__private::Err(__err);
                        }
                    };
                    _serde::ser::SerializeStruct::end(__serde_state)
                }
            }
        };
        impl ::core::marker::StructuralPartialEq for Dense {}
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl ::core::cmp::PartialEq for Dense {
            #[inline]
            fn eq(&self, other: &Dense) -> bool {
                match *other {
                    Dense {
                        aff: ref __self_1_0,
                    } => match *self {
                        Dense {
                            aff: ref __self_0_0,
                        } => (*__self_0_0) == (*__self_1_0),
                    },
                }
            }
            #[inline]
            fn ne(&self, other: &Dense) -> bool {
                match *other {
                    Dense {
                        aff: ref __self_1_0,
                    } => match *self {
                        Dense {
                            aff: ref __self_0_0,
                        } => (*__self_0_0) != (*__self_1_0),
                    },
                }
            }
        }
        impl Dense {
            pub const fn new(aff: Affine2) -> Self {
                Self { aff }
            }
            pub fn from_parts(mul: Array2<NNVFloat>, add: Array1<NNVFloat>) -> Self {
                Self {
                    aff: Affine2::new(mul, add),
                }
            }
        }
        impl Operation for Dense {
            fn as_any(&self) -> &dyn std::any::Any {
                self
            }
            fn input_shapes(&self) -> Vec<TensorShape> {
                <[_]>::into_vec(box [TensorShape::new(<[_]>::into_vec(box [Some(
                    self.aff.input_dim(),
                )]))])
            }
            fn output_shapes(&self) -> Vec<TensorShape> {
                <[_]>::into_vec(box [TensorShape::new(<[_]>::into_vec(box [Some(
                    self.aff.output_dim(),
                )]))])
            }
            fn forward1(&self, input: &[&Array1<NNVFloat>]) -> Vec<Array1<NNVFloat>> {
                if true {
                    match (&input.len(), &1) {
                        (left_val, right_val) => {
                            if !(*left_val == *right_val) {
                                let kind = ::core::panicking::AssertKind::Eq;
                                ::core::panicking::assert_failed(
                                    kind,
                                    &*left_val,
                                    &*right_val,
                                    ::core::option::Option::None,
                                );
                            }
                        }
                    };
                };
                if true {
                    match (&input[0].ndim(), &1) {
                        (left_val, right_val) => {
                            if !(*left_val == *right_val) {
                                let kind = ::core::panicking::AssertKind::Eq;
                                ::core::panicking::assert_failed(
                                    kind,
                                    &*left_val,
                                    &*right_val,
                                    ::core::option::Option::None,
                                );
                            }
                        }
                    };
                };
                <[_]>::into_vec(box [self.aff.apply(&input[0].view())])
            }
            fn forward2(&self, input: &[&Array2<NNVFloat>]) -> Vec<Array2<NNVFloat>> {
                <[_]>::into_vec(box [self.aff.apply_matrix(&input[0].view())])
            }
            fn apply_bounds(
                &self,
                bounds: &[Bounds1],
                lower_aff: &[Affine2],
                upper_aff: &[Affine2],
            ) -> Vec<(Bounds1, Affine2, Affine2)> {
                let new_lower = self.aff.signed_compose(&lower_aff[0], &upper_aff[0]);
                let new_upper = self.aff.signed_compose(&upper_aff[0], &lower_aff[0]);
                <[_]>::into_vec(box [(self.aff.signed_apply(&bounds[0]), new_lower, new_upper)])
            }
            fn forward_star<StarRef: Deref<Target = Star2>>(
                &self,
                stars: Vec<StarRef>,
                _activation_idx: Option<usize>,
                parent_axis_aligned_input_bounds: Vec<&Bounds1>,
            ) -> (Vec<Star2>, Vec<Bounds1>, bool) {
                match (&stars.len(), &1) {
                    (left_val, right_val) => {
                        if !(*left_val == *right_val) {
                            let kind = ::core::panicking::AssertKind::Eq;
                            ::core::panicking::assert_failed(
                                kind,
                                &*left_val,
                                &*right_val,
                                ::core::option::Option::None,
                            );
                        }
                    }
                };
                match (&parent_axis_aligned_input_bounds.len(), &1) {
                    (left_val, right_val) => {
                        if !(*left_val == *right_val) {
                            let kind = ::core::panicking::AssertKind::Eq;
                            ::core::panicking::assert_failed(
                                kind,
                                &*left_val,
                                &*right_val,
                                ::core::option::Option::None,
                            );
                        }
                    }
                };
                (
                    <[_]>::into_vec(box [stars[0].affine_map2(&self.aff)]),
                    <[_]>::into_vec(box [parent_axis_aligned_input_bounds[0].clone()]),
                    false,
                )
            }
        }
        impl fmt::Display for Dense {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_fmt(::core::fmt::Arguments::new_v1(
                    &["Dense "],
                    &[::core::fmt::ArgumentV1::new_display(&self.aff.output_dim())],
                ))
            }
        }
    }
    pub mod dnn {
        use crate::graph::{
            Engine, ExecuteError, Graph, GraphError, Operation, OperationId, PhysicalOp,
            RepresentationId,
        };
        use crate::tensorshape::TensorShape;
        use crate::NNVFloat;
        use ndarray::Array2;
        use ndarray::{Array1, Axis};
        use serde::{Deserialize, Serialize};
        use std::fmt;
        pub struct DNN {
            graph: Graph,
            input_representation_ids: Vec<RepresentationId>,
            output_representation_ids: Vec<RepresentationId>,
        }
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl ::core::clone::Clone for DNN {
            #[inline]
            fn clone(&self) -> DNN {
                match *self {
                    DNN {
                        graph: ref __self_0_0,
                        input_representation_ids: ref __self_0_1,
                        output_representation_ids: ref __self_0_2,
                    } => DNN {
                        graph: ::core::clone::Clone::clone(&(*__self_0_0)),
                        input_representation_ids: ::core::clone::Clone::clone(&(*__self_0_1)),
                        output_representation_ids: ::core::clone::Clone::clone(&(*__self_0_2)),
                    },
                }
            }
        }
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl ::core::default::Default for DNN {
            #[inline]
            fn default() -> DNN {
                DNN {
                    graph: ::core::default::Default::default(),
                    input_representation_ids: ::core::default::Default::default(),
                    output_representation_ids: ::core::default::Default::default(),
                }
            }
        }
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl ::core::fmt::Debug for DNN {
            fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                match *self {
                    DNN {
                        graph: ref __self_0_0,
                        input_representation_ids: ref __self_0_1,
                        output_representation_ids: ref __self_0_2,
                    } => {
                        let debug_trait_builder =
                            &mut ::core::fmt::Formatter::debug_struct(f, "DNN");
                        let _ = ::core::fmt::DebugStruct::field(
                            debug_trait_builder,
                            "graph",
                            &&(*__self_0_0),
                        );
                        let _ = ::core::fmt::DebugStruct::field(
                            debug_trait_builder,
                            "input_representation_ids",
                            &&(*__self_0_1),
                        );
                        let _ = ::core::fmt::DebugStruct::field(
                            debug_trait_builder,
                            "output_representation_ids",
                            &&(*__self_0_2),
                        );
                        ::core::fmt::DebugStruct::finish(debug_trait_builder)
                    }
                }
            }
        }
        impl DNN {
            pub fn new(
                graph: Graph,
                input_representation_ids: Vec<RepresentationId>,
                output_representation_ids: Vec<RepresentationId>,
            ) -> Self {
                Self {
                    graph,
                    input_representation_ids,
                    output_representation_ids,
                }
            }
            /// # Panics
            /// TODO
            pub fn from_sequential(layers: &[PhysicalOp]) -> Self {
                let mut graph = Graph::default();
                for (i, layer) in layers.iter().enumerate() {
                    graph
                        .add_operation(
                            layer.clone(),
                            <[_]>::into_vec(box [RepresentationId::new(i, None)]),
                            <[_]>::into_vec(box [RepresentationId::new(i + 1, None)]),
                        )
                        .unwrap();
                }
                let input_representation_ids =
                    <[_]>::into_vec(box [RepresentationId::new(0, None)]);
                let output_representation_ids =
                    <[_]>::into_vec(box [RepresentationId::new(layers.len(), None)]);
                Self {
                    graph,
                    input_representation_ids,
                    output_representation_ids,
                }
            }
            /// # Errors
            pub fn add_operation(
                &mut self,
                op: PhysicalOp,
                inputs: Vec<RepresentationId>,
                outputs: Vec<RepresentationId>,
            ) -> Result<OperationId, GraphError> {
                self.graph.add_operation(op, inputs, outputs)
            }
            pub fn get_operation(&self, id: OperationId) -> Option<&PhysicalOp> {
                self.graph
                    .get_operation_node(&id)
                    .map(crate::graph::OperationNode::get_operation)
            }
            pub const fn get_input_representation_ids(&self) -> &Vec<RepresentationId> {
                &self.input_representation_ids
            }
            pub const fn get_output_representation_ids(&self) -> &Vec<RepresentationId> {
                &self.output_representation_ids
            }
            pub const fn get_graph(&self) -> &Graph {
                &self.graph
            }
            /// # Returns
            /// `Vec<(num_samples, dimension)>` where each entry is a layer's activations
            ///
            /// # Results
            /// TODO
            ///
            /// # Panics
            /// TODO
            pub fn calculate_activation_pattern2(
                &self,
                inputs: &[Array2<NNVFloat>],
            ) -> Result<Vec<Array2<bool>>, ExecuteError> {
                match (&inputs.len(), &self.get_input_representation_ids().len()) {
                    (left_val, right_val) => {
                        if !(*left_val == *right_val) {
                            let kind = ::core::panicking::AssertKind::Eq;
                            ::core::panicking::assert_failed(
                                kind,
                                &*left_val,
                                &*right_val,
                                ::core::option::Option::None,
                            );
                        }
                    }
                };
                let mut activation_patterns = ::alloc::vec::Vec::new();
                let inputs = self
                    .get_input_representation_ids()
                    .iter()
                    .zip(inputs.iter())
                    .map(|(&id, input)| (id, input.clone()))
                    .collect::<Vec<_>>();
                Engine::new(&self.graph).run(
                    self.get_output_representation_ids().clone(),
                    &inputs,
                    |op, inputs, _| -> (Option<usize>, Vec<Array2<NNVFloat>>) {
                        if let Some(mut pattern) = op.get_activation_pattern(inputs) {
                            activation_patterns.append(&mut pattern);
                        }
                        (None, op.forward2(inputs))
                    },
                )?;
                Ok(activation_patterns)
            }
            pub fn calculate_activation_pattern1(
                &self,
                inputs: &[Array1<NNVFloat>],
            ) -> Result<Vec<Array1<bool>>, ExecuteError> {
                Ok(self
                    .calculate_activation_pattern2(
                        &inputs
                            .iter()
                            .map(|input| input.clone().insert_axis(Axis(1)))
                            .collect::<Vec<_>>(),
                    )?
                    .into_iter()
                    .map(|x| x.index_axis(Axis(1), 0).to_owned())
                    .collect())
            }
            /// # Panics
            /// TODO
            pub fn forward1(&self, inputs: &[Array1<NNVFloat>]) -> Vec<Array1<NNVFloat>> {
                match (&inputs.len(), &self.get_input_representation_ids().len()) {
                    (left_val, right_val) => {
                        if !(*left_val == *right_val) {
                            let kind = ::core::panicking::AssertKind::Eq;
                            ::core::panicking::assert_failed(
                                kind,
                                &*left_val,
                                &*right_val,
                                ::core::option::Option::None,
                            );
                        }
                    }
                };
                let engine = Engine::new(&self.graph);
                let inputs = self
                    .get_input_representation_ids()
                    .iter()
                    .zip(inputs.iter())
                    .map(|(&id, input)| (id, input.clone()))
                    .collect::<Vec<_>>();
                let res = engine.run(
                    self.get_output_representation_ids().clone(),
                    &inputs,
                    |op: &dyn Operation, inputs, _| -> (Option<usize>, Vec<Array1<NNVFloat>>) {
                        (None, op.forward1(inputs))
                    },
                );
                res.unwrap().into_iter().map(|(_, output)| output).collect()
            }
            /// # Panics
            /// TODO
            pub fn forward_suffix1(
                &self,
                inputs: &[(RepresentationId, Array1<NNVFloat>)],
            ) -> Vec<Array1<NNVFloat>> {
                let engine = Engine::new(&self.graph);
                let res = engine.run(
                    self.get_output_representation_ids().clone(),
                    &inputs,
                    |op: &dyn Operation, inputs, _| -> (Option<usize>, Vec<Array1<NNVFloat>>) {
                        (None, op.forward1(inputs))
                    },
                );
                res.unwrap().into_iter().map(|(_, output)| output).collect()
            }
        }
        impl DNN {
            /// # Panics
            /// TODO
            pub fn input_shapes(&self) -> Vec<TensorShape> {
                self.input_representation_ids
                    .iter()
                    .map(|repr_id| {
                        let op_id = *self
                            .graph
                            .get_representation_input_op_ids(repr_id)
                            .first()
                            .unwrap();
                        let op_node = self.graph.get_operation_node(&op_id).unwrap();
                        let idx = op_node
                            .get_input_ids()
                            .iter()
                            .position(|x| *x == *repr_id)
                            .unwrap();
                        op_node.get_operation().input_shapes()[idx].clone()
                    })
                    .collect::<Vec<_>>()
            }
            /// # Panics
            pub fn output_shapes(&self) -> Vec<TensorShape> {
                self.output_representation_ids
                    .iter()
                    .map(|repr_id| {
                        let op_id = self.graph.get_representation_op_id(repr_id).unwrap();
                        let op_node = self.graph.get_operation_node(&op_id).unwrap();
                        let idx = op_node
                            .get_input_ids()
                            .iter()
                            .position(|x| *x == *repr_id)
                            .unwrap();
                        op_node.get_operation().input_shapes()[idx].clone()
                    })
                    .collect::<Vec<_>>()
            }
        }
        impl fmt::Display for DNN {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_fmt(::core::fmt::Arguments::new_v1(
                    &["Input ", " => "],
                    &[
                        ::core::fmt::ArgumentV1::new_debug(&self.input_shapes()),
                        ::core::fmt::ArgumentV1::new_debug(&self.output_shapes()),
                    ],
                ))
            }
        }
    }
    pub mod interpolate {
        #![allow(
            non_snake_case,
            clippy::module_name_repetitions,
            clippy::cast_precision_loss,
            clippy::cast_sign_loss,
            clippy::cast_possible_truncation
        )]
        //! Representation of affine transformations
        use crate::affine::Affine2;
        use crate::bounds::Bounds1;
        use crate::graph::Operation;
        use crate::star::Star2;
        use crate::tensorshape::TensorShape;
        use crate::NNVFloat;
        use itertools::Itertools;
        use ndarray::{Array1, Array2};
        use serde::{Deserialize, Serialize};
        use std::any::Any;
        use std::fmt;
        use std::fmt::Debug;
        use std::ops::Deref;
        pub enum InterpolateMethod {
            Bilinear,
        }
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl ::core::clone::Clone for InterpolateMethod {
            #[inline]
            fn clone(&self) -> InterpolateMethod {
                match (&*self,) {
                    (&InterpolateMethod::Bilinear,) => InterpolateMethod::Bilinear,
                }
            }
        }
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl ::core::fmt::Debug for InterpolateMethod {
            fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                match (&*self,) {
                    (&InterpolateMethod::Bilinear,) => {
                        ::core::fmt::Formatter::write_str(f, "Bilinear")
                    }
                }
            }
        }
        #[doc(hidden)]
        #[allow(non_upper_case_globals, unused_attributes, unused_qualifications)]
        const _: () = {
            #[allow(unused_extern_crates, clippy::useless_attribute)]
            extern crate serde as _serde;
            #[automatically_derived]
            impl<'de> _serde::Deserialize<'de> for InterpolateMethod {
                fn deserialize<__D>(
                    __deserializer: __D,
                ) -> _serde::__private::Result<Self, __D::Error>
                where
                    __D: _serde::Deserializer<'de>,
                {
                    #[allow(non_camel_case_types)]
                    enum __Field {
                        __field0,
                    }
                    struct __FieldVisitor;
                    impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                        type Value = __Field;
                        fn expecting(
                            &self,
                            __formatter: &mut _serde::__private::Formatter,
                        ) -> _serde::__private::fmt::Result {
                            _serde::__private::Formatter::write_str(
                                __formatter,
                                "variant identifier",
                            )
                        }
                        fn visit_u64<__E>(
                            self,
                            __value: u64,
                        ) -> _serde::__private::Result<Self::Value, __E>
                        where
                            __E: _serde::de::Error,
                        {
                            match __value {
                                0u64 => _serde::__private::Ok(__Field::__field0),
                                _ => _serde::__private::Err(_serde::de::Error::invalid_value(
                                    _serde::de::Unexpected::Unsigned(__value),
                                    &"variant index 0 <= i < 1",
                                )),
                            }
                        }
                        fn visit_str<__E>(
                            self,
                            __value: &str,
                        ) -> _serde::__private::Result<Self::Value, __E>
                        where
                            __E: _serde::de::Error,
                        {
                            match __value {
                                "Bilinear" => _serde::__private::Ok(__Field::__field0),
                                _ => _serde::__private::Err(_serde::de::Error::unknown_variant(
                                    __value, VARIANTS,
                                )),
                            }
                        }
                        fn visit_bytes<__E>(
                            self,
                            __value: &[u8],
                        ) -> _serde::__private::Result<Self::Value, __E>
                        where
                            __E: _serde::de::Error,
                        {
                            match __value {
                                b"Bilinear" => _serde::__private::Ok(__Field::__field0),
                                _ => {
                                    let __value = &_serde::__private::from_utf8_lossy(__value);
                                    _serde::__private::Err(_serde::de::Error::unknown_variant(
                                        __value, VARIANTS,
                                    ))
                                }
                            }
                        }
                    }
                    impl<'de> _serde::Deserialize<'de> for __Field {
                        #[inline]
                        fn deserialize<__D>(
                            __deserializer: __D,
                        ) -> _serde::__private::Result<Self, __D::Error>
                        where
                            __D: _serde::Deserializer<'de>,
                        {
                            _serde::Deserializer::deserialize_identifier(
                                __deserializer,
                                __FieldVisitor,
                            )
                        }
                    }
                    struct __Visitor<'de> {
                        marker: _serde::__private::PhantomData<InterpolateMethod>,
                        lifetime: _serde::__private::PhantomData<&'de ()>,
                    }
                    impl<'de> _serde::de::Visitor<'de> for __Visitor<'de> {
                        type Value = InterpolateMethod;
                        fn expecting(
                            &self,
                            __formatter: &mut _serde::__private::Formatter,
                        ) -> _serde::__private::fmt::Result {
                            _serde::__private::Formatter::write_str(
                                __formatter,
                                "enum InterpolateMethod",
                            )
                        }
                        fn visit_enum<__A>(
                            self,
                            __data: __A,
                        ) -> _serde::__private::Result<Self::Value, __A::Error>
                        where
                            __A: _serde::de::EnumAccess<'de>,
                        {
                            match match _serde::de::EnumAccess::variant(__data) {
                                _serde::__private::Ok(__val) => __val,
                                _serde::__private::Err(__err) => {
                                    return _serde::__private::Err(__err);
                                }
                            } {
                                (__Field::__field0, __variant) => {
                                    match _serde::de::VariantAccess::unit_variant(__variant) {
                                        _serde::__private::Ok(__val) => __val,
                                        _serde::__private::Err(__err) => {
                                            return _serde::__private::Err(__err);
                                        }
                                    };
                                    _serde::__private::Ok(InterpolateMethod::Bilinear)
                                }
                            }
                        }
                    }
                    const VARIANTS: &'static [&'static str] = &["Bilinear"];
                    _serde::Deserializer::deserialize_enum(
                        __deserializer,
                        "InterpolateMethod",
                        VARIANTS,
                        __Visitor {
                            marker: _serde::__private::PhantomData::<InterpolateMethod>,
                            lifetime: _serde::__private::PhantomData,
                        },
                    )
                }
            }
        };
        #[doc(hidden)]
        #[allow(non_upper_case_globals, unused_attributes, unused_qualifications)]
        const _: () = {
            #[allow(unused_extern_crates, clippy::useless_attribute)]
            extern crate serde as _serde;
            #[automatically_derived]
            impl _serde::Serialize for InterpolateMethod {
                fn serialize<__S>(
                    &self,
                    __serializer: __S,
                ) -> _serde::__private::Result<__S::Ok, __S::Error>
                where
                    __S: _serde::Serializer,
                {
                    match *self {
                        InterpolateMethod::Bilinear => _serde::Serializer::serialize_unit_variant(
                            __serializer,
                            "InterpolateMethod",
                            0u32,
                            "Bilinear",
                        ),
                    }
                }
            }
        };
        /// Assumes that data is always in a flattened state.
        /// Weights are of the shape: (`kernel_w`, `kernel_h`, `channels_in`, `channels_out`)
        pub struct Interpolate {
            input_shape: TensorShape,
            output_shape: TensorShape,
            method: InterpolateMethod,
            affine: Option<Affine2>,
        }
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl ::core::clone::Clone for Interpolate {
            #[inline]
            fn clone(&self) -> Interpolate {
                match *self {
                    Interpolate {
                        input_shape: ref __self_0_0,
                        output_shape: ref __self_0_1,
                        method: ref __self_0_2,
                        affine: ref __self_0_3,
                    } => Interpolate {
                        input_shape: ::core::clone::Clone::clone(&(*__self_0_0)),
                        output_shape: ::core::clone::Clone::clone(&(*__self_0_1)),
                        method: ::core::clone::Clone::clone(&(*__self_0_2)),
                        affine: ::core::clone::Clone::clone(&(*__self_0_3)),
                    },
                }
            }
        }
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl ::core::fmt::Debug for Interpolate {
            fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                match *self {
                    Interpolate {
                        input_shape: ref __self_0_0,
                        output_shape: ref __self_0_1,
                        method: ref __self_0_2,
                        affine: ref __self_0_3,
                    } => {
                        let debug_trait_builder =
                            &mut ::core::fmt::Formatter::debug_struct(f, "Interpolate");
                        let _ = ::core::fmt::DebugStruct::field(
                            debug_trait_builder,
                            "input_shape",
                            &&(*__self_0_0),
                        );
                        let _ = ::core::fmt::DebugStruct::field(
                            debug_trait_builder,
                            "output_shape",
                            &&(*__self_0_1),
                        );
                        let _ = ::core::fmt::DebugStruct::field(
                            debug_trait_builder,
                            "method",
                            &&(*__self_0_2),
                        );
                        let _ = ::core::fmt::DebugStruct::field(
                            debug_trait_builder,
                            "affine",
                            &&(*__self_0_3),
                        );
                        ::core::fmt::DebugStruct::finish(debug_trait_builder)
                    }
                }
            }
        }
        #[doc(hidden)]
        #[allow(non_upper_case_globals, unused_attributes, unused_qualifications)]
        const _: () = {
            #[allow(unused_extern_crates, clippy::useless_attribute)]
            extern crate serde as _serde;
            #[automatically_derived]
            impl<'de> _serde::Deserialize<'de> for Interpolate {
                fn deserialize<__D>(
                    __deserializer: __D,
                ) -> _serde::__private::Result<Self, __D::Error>
                where
                    __D: _serde::Deserializer<'de>,
                {
                    #[allow(non_camel_case_types)]
                    enum __Field {
                        __field0,
                        __field1,
                        __field2,
                        __field3,
                        __ignore,
                    }
                    struct __FieldVisitor;
                    impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                        type Value = __Field;
                        fn expecting(
                            &self,
                            __formatter: &mut _serde::__private::Formatter,
                        ) -> _serde::__private::fmt::Result {
                            _serde::__private::Formatter::write_str(__formatter, "field identifier")
                        }
                        fn visit_u64<__E>(
                            self,
                            __value: u64,
                        ) -> _serde::__private::Result<Self::Value, __E>
                        where
                            __E: _serde::de::Error,
                        {
                            match __value {
                                0u64 => _serde::__private::Ok(__Field::__field0),
                                1u64 => _serde::__private::Ok(__Field::__field1),
                                2u64 => _serde::__private::Ok(__Field::__field2),
                                3u64 => _serde::__private::Ok(__Field::__field3),
                                _ => _serde::__private::Ok(__Field::__ignore),
                            }
                        }
                        fn visit_str<__E>(
                            self,
                            __value: &str,
                        ) -> _serde::__private::Result<Self::Value, __E>
                        where
                            __E: _serde::de::Error,
                        {
                            match __value {
                                "input_shape" => _serde::__private::Ok(__Field::__field0),
                                "output_shape" => _serde::__private::Ok(__Field::__field1),
                                "method" => _serde::__private::Ok(__Field::__field2),
                                "affine" => _serde::__private::Ok(__Field::__field3),
                                _ => _serde::__private::Ok(__Field::__ignore),
                            }
                        }
                        fn visit_bytes<__E>(
                            self,
                            __value: &[u8],
                        ) -> _serde::__private::Result<Self::Value, __E>
                        where
                            __E: _serde::de::Error,
                        {
                            match __value {
                                b"input_shape" => _serde::__private::Ok(__Field::__field0),
                                b"output_shape" => _serde::__private::Ok(__Field::__field1),
                                b"method" => _serde::__private::Ok(__Field::__field2),
                                b"affine" => _serde::__private::Ok(__Field::__field3),
                                _ => _serde::__private::Ok(__Field::__ignore),
                            }
                        }
                    }
                    impl<'de> _serde::Deserialize<'de> for __Field {
                        #[inline]
                        fn deserialize<__D>(
                            __deserializer: __D,
                        ) -> _serde::__private::Result<Self, __D::Error>
                        where
                            __D: _serde::Deserializer<'de>,
                        {
                            _serde::Deserializer::deserialize_identifier(
                                __deserializer,
                                __FieldVisitor,
                            )
                        }
                    }
                    struct __Visitor<'de> {
                        marker: _serde::__private::PhantomData<Interpolate>,
                        lifetime: _serde::__private::PhantomData<&'de ()>,
                    }
                    impl<'de> _serde::de::Visitor<'de> for __Visitor<'de> {
                        type Value = Interpolate;
                        fn expecting(
                            &self,
                            __formatter: &mut _serde::__private::Formatter,
                        ) -> _serde::__private::fmt::Result {
                            _serde::__private::Formatter::write_str(
                                __formatter,
                                "struct Interpolate",
                            )
                        }
                        #[inline]
                        fn visit_seq<__A>(
                            self,
                            mut __seq: __A,
                        ) -> _serde::__private::Result<Self::Value, __A::Error>
                        where
                            __A: _serde::de::SeqAccess<'de>,
                        {
                            let __field0 = match match _serde::de::SeqAccess::next_element::<
                                TensorShape,
                            >(&mut __seq)
                            {
                                _serde::__private::Ok(__val) => __val,
                                _serde::__private::Err(__err) => {
                                    return _serde::__private::Err(__err);
                                }
                            } {
                                _serde::__private::Some(__value) => __value,
                                _serde::__private::None => {
                                    return _serde::__private::Err(
                                        _serde::de::Error::invalid_length(
                                            0usize,
                                            &"struct Interpolate with 4 elements",
                                        ),
                                    );
                                }
                            };
                            let __field1 = match match _serde::de::SeqAccess::next_element::<
                                TensorShape,
                            >(&mut __seq)
                            {
                                _serde::__private::Ok(__val) => __val,
                                _serde::__private::Err(__err) => {
                                    return _serde::__private::Err(__err);
                                }
                            } {
                                _serde::__private::Some(__value) => __value,
                                _serde::__private::None => {
                                    return _serde::__private::Err(
                                        _serde::de::Error::invalid_length(
                                            1usize,
                                            &"struct Interpolate with 4 elements",
                                        ),
                                    );
                                }
                            };
                            let __field2 = match match _serde::de::SeqAccess::next_element::<
                                InterpolateMethod,
                            >(&mut __seq)
                            {
                                _serde::__private::Ok(__val) => __val,
                                _serde::__private::Err(__err) => {
                                    return _serde::__private::Err(__err);
                                }
                            } {
                                _serde::__private::Some(__value) => __value,
                                _serde::__private::None => {
                                    return _serde::__private::Err(
                                        _serde::de::Error::invalid_length(
                                            2usize,
                                            &"struct Interpolate with 4 elements",
                                        ),
                                    );
                                }
                            };
                            let __field3 = match match _serde::de::SeqAccess::next_element::<
                                Option<Affine2>,
                            >(&mut __seq)
                            {
                                _serde::__private::Ok(__val) => __val,
                                _serde::__private::Err(__err) => {
                                    return _serde::__private::Err(__err);
                                }
                            } {
                                _serde::__private::Some(__value) => __value,
                                _serde::__private::None => {
                                    return _serde::__private::Err(
                                        _serde::de::Error::invalid_length(
                                            3usize,
                                            &"struct Interpolate with 4 elements",
                                        ),
                                    );
                                }
                            };
                            _serde::__private::Ok(Interpolate {
                                input_shape: __field0,
                                output_shape: __field1,
                                method: __field2,
                                affine: __field3,
                            })
                        }
                        #[inline]
                        fn visit_map<__A>(
                            self,
                            mut __map: __A,
                        ) -> _serde::__private::Result<Self::Value, __A::Error>
                        where
                            __A: _serde::de::MapAccess<'de>,
                        {
                            let mut __field0: _serde::__private::Option<TensorShape> =
                                _serde::__private::None;
                            let mut __field1: _serde::__private::Option<TensorShape> =
                                _serde::__private::None;
                            let mut __field2: _serde::__private::Option<InterpolateMethod> =
                                _serde::__private::None;
                            let mut __field3: _serde::__private::Option<Option<Affine2>> =
                                _serde::__private::None;
                            while let _serde::__private::Some(__key) =
                                match _serde::de::MapAccess::next_key::<__Field>(&mut __map) {
                                    _serde::__private::Ok(__val) => __val,
                                    _serde::__private::Err(__err) => {
                                        return _serde::__private::Err(__err);
                                    }
                                }
                            {
                                match __key {
                                    __Field::__field0 => {
                                        if _serde::__private::Option::is_some(&__field0) {
                                            return _serde::__private::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field(
                                                    "input_shape",
                                                ),
                                            );
                                        }
                                        __field0 = _serde::__private::Some(
                                            match _serde::de::MapAccess::next_value::<TensorShape>(
                                                &mut __map,
                                            ) {
                                                _serde::__private::Ok(__val) => __val,
                                                _serde::__private::Err(__err) => {
                                                    return _serde::__private::Err(__err);
                                                }
                                            },
                                        );
                                    }
                                    __Field::__field1 => {
                                        if _serde::__private::Option::is_some(&__field1) {
                                            return _serde::__private::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field(
                                                    "output_shape",
                                                ),
                                            );
                                        }
                                        __field1 = _serde::__private::Some(
                                            match _serde::de::MapAccess::next_value::<TensorShape>(
                                                &mut __map,
                                            ) {
                                                _serde::__private::Ok(__val) => __val,
                                                _serde::__private::Err(__err) => {
                                                    return _serde::__private::Err(__err);
                                                }
                                            },
                                        );
                                    }
                                    __Field::__field2 => {
                                        if _serde::__private::Option::is_some(&__field2) {
                                            return _serde::__private::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field(
                                                    "method",
                                                ),
                                            );
                                        }
                                        __field2 = _serde::__private::Some(
                                            match _serde::de::MapAccess::next_value::<
                                                InterpolateMethod,
                                            >(
                                                &mut __map
                                            ) {
                                                _serde::__private::Ok(__val) => __val,
                                                _serde::__private::Err(__err) => {
                                                    return _serde::__private::Err(__err);
                                                }
                                            },
                                        );
                                    }
                                    __Field::__field3 => {
                                        if _serde::__private::Option::is_some(&__field3) {
                                            return _serde::__private::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field(
                                                    "affine",
                                                ),
                                            );
                                        }
                                        __field3 = _serde::__private::Some(
                                            match _serde::de::MapAccess::next_value::<Option<Affine2>>(
                                                &mut __map,
                                            ) {
                                                _serde::__private::Ok(__val) => __val,
                                                _serde::__private::Err(__err) => {
                                                    return _serde::__private::Err(__err);
                                                }
                                            },
                                        );
                                    }
                                    _ => {
                                        let _ = match _serde::de::MapAccess::next_value::<
                                            _serde::de::IgnoredAny,
                                        >(
                                            &mut __map
                                        ) {
                                            _serde::__private::Ok(__val) => __val,
                                            _serde::__private::Err(__err) => {
                                                return _serde::__private::Err(__err);
                                            }
                                        };
                                    }
                                }
                            }
                            let __field0 = match __field0 {
                                _serde::__private::Some(__field0) => __field0,
                                _serde::__private::None => {
                                    match _serde::__private::de::missing_field("input_shape") {
                                        _serde::__private::Ok(__val) => __val,
                                        _serde::__private::Err(__err) => {
                                            return _serde::__private::Err(__err);
                                        }
                                    }
                                }
                            };
                            let __field1 = match __field1 {
                                _serde::__private::Some(__field1) => __field1,
                                _serde::__private::None => {
                                    match _serde::__private::de::missing_field("output_shape") {
                                        _serde::__private::Ok(__val) => __val,
                                        _serde::__private::Err(__err) => {
                                            return _serde::__private::Err(__err);
                                        }
                                    }
                                }
                            };
                            let __field2 = match __field2 {
                                _serde::__private::Some(__field2) => __field2,
                                _serde::__private::None => {
                                    match _serde::__private::de::missing_field("method") {
                                        _serde::__private::Ok(__val) => __val,
                                        _serde::__private::Err(__err) => {
                                            return _serde::__private::Err(__err);
                                        }
                                    }
                                }
                            };
                            let __field3 = match __field3 {
                                _serde::__private::Some(__field3) => __field3,
                                _serde::__private::None => {
                                    match _serde::__private::de::missing_field("affine") {
                                        _serde::__private::Ok(__val) => __val,
                                        _serde::__private::Err(__err) => {
                                            return _serde::__private::Err(__err);
                                        }
                                    }
                                }
                            };
                            _serde::__private::Ok(Interpolate {
                                input_shape: __field0,
                                output_shape: __field1,
                                method: __field2,
                                affine: __field3,
                            })
                        }
                    }
                    const FIELDS: &'static [&'static str] =
                        &["input_shape", "output_shape", "method", "affine"];
                    _serde::Deserializer::deserialize_struct(
                        __deserializer,
                        "Interpolate",
                        FIELDS,
                        __Visitor {
                            marker: _serde::__private::PhantomData::<Interpolate>,
                            lifetime: _serde::__private::PhantomData,
                        },
                    )
                }
            }
        };
        #[doc(hidden)]
        #[allow(non_upper_case_globals, unused_attributes, unused_qualifications)]
        const _: () = {
            #[allow(unused_extern_crates, clippy::useless_attribute)]
            extern crate serde as _serde;
            #[automatically_derived]
            impl _serde::Serialize for Interpolate {
                fn serialize<__S>(
                    &self,
                    __serializer: __S,
                ) -> _serde::__private::Result<__S::Ok, __S::Error>
                where
                    __S: _serde::Serializer,
                {
                    let mut __serde_state = match _serde::Serializer::serialize_struct(
                        __serializer,
                        "Interpolate",
                        false as usize + 1 + 1 + 1 + 1,
                    ) {
                        _serde::__private::Ok(__val) => __val,
                        _serde::__private::Err(__err) => {
                            return _serde::__private::Err(__err);
                        }
                    };
                    match _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "input_shape",
                        &self.input_shape,
                    ) {
                        _serde::__private::Ok(__val) => __val,
                        _serde::__private::Err(__err) => {
                            return _serde::__private::Err(__err);
                        }
                    };
                    match _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "output_shape",
                        &self.output_shape,
                    ) {
                        _serde::__private::Ok(__val) => __val,
                        _serde::__private::Err(__err) => {
                            return _serde::__private::Err(__err);
                        }
                    };
                    match _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "method",
                        &self.method,
                    ) {
                        _serde::__private::Ok(__val) => __val,
                        _serde::__private::Err(__err) => {
                            return _serde::__private::Err(__err);
                        }
                    };
                    match _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "affine",
                        &self.affine,
                    ) {
                        _serde::__private::Ok(__val) => __val,
                        _serde::__private::Err(__err) => {
                            return _serde::__private::Err(__err);
                        }
                    };
                    _serde::ser::SerializeStruct::end(__serde_state)
                }
            }
        };
        impl Interpolate {
            /// # Panics
            /// If improper shapes are passed in
            pub fn new(
                input_shape: TensorShape,
                output_shape: TensorShape,
                method: InterpolateMethod,
            ) -> Self {
                match (&input_shape[3].unwrap(), &output_shape[3].unwrap()) {
                    (left_val, right_val) => {
                        if !(*left_val == *right_val) {
                            let kind = ::core::panicking::AssertKind::Eq;
                            ::core::panicking::assert_failed(
                                kind,
                                &*left_val,
                                &*right_val,
                                ::core::option::Option::None,
                            );
                        }
                    }
                };
                let mut s = Self {
                    input_shape,
                    output_shape,
                    method,
                    affine: None,
                };
                s.construct_affine();
                s
            }
            pub fn get_affine(&self) -> &Affine2 {
                self.affine.as_ref().unwrap()
            }
            pub fn input_shape(&self) -> TensorShape {
                self.input_shape.clone()
            }
            pub fn output_shape(&self) -> TensorShape {
                self.output_shape.clone()
            }
            fn construct_affine(&mut self) {
                let h_in = self.input_shape[1].unwrap();
                let w_in = self.input_shape[2].unwrap();
                let c = self.input_shape[3].unwrap();
                let h_out = self.output_shape()[1].unwrap();
                let w_out = self.output_shape()[2].unwrap();
                if true {
                    if !(h_in > 0) {
                        ::core::panicking::panic("assertion failed: h_in > 0")
                    };
                };
                if true {
                    if !(w_in > 0) {
                        ::core::panicking::panic("assertion failed: w_in > 0")
                    };
                };
                if true {
                    if !(c > 0) {
                        ::core::panicking::panic("assertion failed: c > 0")
                    };
                };
                if true {
                    if !(h_out > 0) {
                        ::core::panicking::panic("assertion failed: h_out > 0")
                    };
                };
                if true {
                    if !(w_out > 0) {
                        ::core::panicking::panic("assertion failed: w_out > 0")
                    };
                };
                let input_dims = h_in * w_in * c;
                let output_dims = h_out * w_out * c;
                let mut weight = Array2::<NNVFloat>::zeros((output_dims, input_dims));
                for ((y_out, x_out), c_out) in (0..h_out)
                    .cartesian_product(0..w_out)
                    .cartesian_product(0..c)
                {
                    let mut x_1: usize = 0;
                    let mut x_2: usize = 0;
                    let mut y_1: usize = 0;
                    let mut y_2: usize = 0;
                    if w_out > 1 {
                        x_1 = (x_out as f64 * (w_in - 1) as f64 / (w_out - 1) as f64).floor()
                            as usize;
                        x_2 =
                            (x_out as f64 * (w_in - 1) as f64 / (w_out - 1) as f64).ceil() as usize;
                    }
                    if h_out > 1 {
                        y_1 = (y_out as f64 * (h_in - 1) as f64 / (h_out - 1) as f64).floor()
                            as usize;
                        y_2 =
                            (y_out as f64 * (h_in - 1) as f64 / (h_out - 1) as f64).ceil() as usize;
                    }
                    let output_idx = y_out * (w_out * c) + x_out * c + c_out;
                    let input_idx_11 = y_1 * (w_in * c) + x_1 * c + c_out;
                    if x_1 == x_2 && y_1 == y_2 {
                        weight[[output_idx, input_idx_11]] = 1.;
                    } else if x_1 == x_2 {
                        let input_idx_2 = y_2 * (w_in * c) + x_2 * c + c_out;
                        let prop_width = x_2 as f64 / w_out as f64 - x_1 as f64 / w_in as f64;
                        let weight_1 =
                            (x_out as f64 / w_out as f64 - x_1 as f64 / w_in as f64) / prop_width;
                        let weight_2 =
                            (x_2 as f64 / w_out as f64 - x_out as f64 / w_in as f64) / prop_width;
                        weight[[output_idx, input_idx_11]] = weight_1;
                        weight[[output_idx, input_idx_2]] = weight_2;
                    } else if y_1 == y_2 {
                        let input_idx_2 = y_2 * (w_in * c) + x_2 * c + c_out;
                        let prop_height = y_2 as f64 / h_out as f64 - y_1 as f64 / h_in as f64;
                        let weight_1 =
                            (y_out as f64 / h_out as f64 - y_1 as f64 / h_in as f64) / prop_height;
                        let weight_2 =
                            (y_2 as f64 / h_out as f64 - y_out as f64 / h_in as f64) / prop_height;
                        weight[[output_idx, input_idx_11]] = weight_1;
                        weight[[output_idx, input_idx_2]] = weight_2;
                    } else {
                        let input_idx_12 = y_1 * (w_in * c) + x_2 * c + c_out;
                        let input_idx_21 = y_2 * (w_in * c) + x_1 * c + c_out;
                        let input_idx_22 = y_2 * (w_in * c) + x_2 * c + c_out;
                        let prop_width = x_2 as f64 / w_out as f64 - x_1 as f64 / w_in as f64;
                        let weight_x_1 =
                            (x_out as f64 / w_out as f64 - x_1 as f64 / w_in as f64) / prop_width;
                        let weight_x_2 =
                            (x_2 as f64 / w_out as f64 - x_out as f64 / w_in as f64) / prop_width;
                        let prop_height = y_2 as f64 / h_out as f64 - y_1 as f64 / h_in as f64;
                        let weight_y_1 =
                            (y_out as f64 / h_out as f64 - y_1 as f64 / h_in as f64) / prop_height;
                        let weight_y_2 =
                            (y_2 as f64 / h_out as f64 - y_out as f64 / h_in as f64) / prop_height;
                        weight[[output_idx, input_idx_11]] = weight_y_1 * weight_x_1;
                        weight[[output_idx, input_idx_12]] = weight_y_1 * weight_x_2;
                        weight[[output_idx, input_idx_21]] = weight_y_2 * weight_x_1;
                        weight[[output_idx, input_idx_22]] = weight_y_2 * weight_x_2;
                    }
                }
                let bias = Array1::<NNVFloat>::zeros(h_out * w_out * c);
                self.affine = Some(Affine2::new(weight, bias));
            }
        }
        impl Operation for Interpolate {
            fn as_any(&self) -> &dyn Any {
                self
            }
            fn input_shapes(&self) -> Vec<TensorShape> {
                <[_]>::into_vec(box [TensorShape::new(<[_]>::into_vec(box [Some(
                    self.get_affine().input_dim(),
                )]))])
            }
            fn output_shapes(&self) -> Vec<TensorShape> {
                <[_]>::into_vec(box [TensorShape::new(<[_]>::into_vec(box [Some(
                    self.get_affine().output_dim(),
                )]))])
            }
            fn forward1(&self, input: &[&Array1<NNVFloat>]) -> Vec<Array1<NNVFloat>> {
                match (&input.len(), &1) {
                    (left_val, right_val) => {
                        if !(*left_val == *right_val) {
                            let kind = ::core::panicking::AssertKind::Eq;
                            ::core::panicking::assert_failed(
                                kind,
                                &*left_val,
                                &*right_val,
                                ::core::option::Option::None,
                            );
                        }
                    }
                };
                let input = input.first().unwrap();
                if true {
                    match (&input.ndim(), &1) {
                        (left_val, right_val) => {
                            if !(*left_val == *right_val) {
                                let kind = ::core::panicking::AssertKind::Eq;
                                ::core::panicking::assert_failed(
                                    kind,
                                    &*left_val,
                                    &*right_val,
                                    ::core::option::Option::None,
                                );
                            }
                        }
                    };
                };
                <[_]>::into_vec(box [self.get_affine().apply(&input.view())])
            }
            fn forward2(&self, input: &[&Array2<NNVFloat>]) -> Vec<Array2<NNVFloat>> {
                match (&input.len(), &1) {
                    (left_val, right_val) => {
                        if !(*left_val == *right_val) {
                            let kind = ::core::panicking::AssertKind::Eq;
                            ::core::panicking::assert_failed(
                                kind,
                                &*left_val,
                                &*right_val,
                                ::core::option::Option::None,
                            );
                        }
                    }
                };
                let input = input.first().unwrap();
                <[_]>::into_vec(box [self.get_affine().apply_matrix(&input.view())])
            }
            fn apply_bounds(
                &self,
                bounds: &[Bounds1],
                lower_aff: &[Affine2],
                upper_aff: &[Affine2],
            ) -> Vec<(Bounds1, Affine2, Affine2)> {
                match (&bounds.len(), &1) {
                    (left_val, right_val) => {
                        if !(*left_val == *right_val) {
                            let kind = ::core::panicking::AssertKind::Eq;
                            ::core::panicking::assert_failed(
                                kind,
                                &*left_val,
                                &*right_val,
                                ::core::option::Option::None,
                            );
                        }
                    }
                };
                match (&lower_aff.len(), &1) {
                    (left_val, right_val) => {
                        if !(*left_val == *right_val) {
                            let kind = ::core::panicking::AssertKind::Eq;
                            ::core::panicking::assert_failed(
                                kind,
                                &*left_val,
                                &*right_val,
                                ::core::option::Option::None,
                            );
                        }
                    }
                };
                match (&upper_aff.len(), &1) {
                    (left_val, right_val) => {
                        if !(*left_val == *right_val) {
                            let kind = ::core::panicking::AssertKind::Eq;
                            ::core::panicking::assert_failed(
                                kind,
                                &*left_val,
                                &*right_val,
                                ::core::option::Option::None,
                            );
                        }
                    }
                };
                let bounds = bounds.first().unwrap();
                let lower_aff = lower_aff.first().unwrap();
                let upper_aff = upper_aff.first().unwrap();
                let new_lower = self.get_affine().signed_compose(lower_aff, upper_aff);
                let new_upper = self.get_affine().signed_compose(upper_aff, lower_aff);
                <[_]>::into_vec(box [(
                    self.get_affine().signed_apply(bounds),
                    new_lower,
                    new_upper,
                )])
            }
            fn forward_star<StarRef: Deref<Target = Star2>>(
                &self,
                stars: Vec<StarRef>,
                _activation_idx: Option<usize>,
                parent_axis_aligned_input_bounds: Vec<&Bounds1>,
            ) -> (Vec<Star2>, Vec<Bounds1>, bool) {
                match (&1, &stars.len()) {
                    (left_val, right_val) => {
                        if !(*left_val == *right_val) {
                            let kind = ::core::panicking::AssertKind::Eq;
                            ::core::panicking::assert_failed(
                                kind,
                                &*left_val,
                                &*right_val,
                                ::core::option::Option::None,
                            );
                        }
                    }
                };
                match (&parent_axis_aligned_input_bounds.len(), &1) {
                    (left_val, right_val) => {
                        if !(*left_val == *right_val) {
                            let kind = ::core::panicking::AssertKind::Eq;
                            ::core::panicking::assert_failed(
                                kind,
                                &*left_val,
                                &*right_val,
                                ::core::option::Option::None,
                            );
                        }
                    }
                };
                (
                    <[_]>::into_vec(box [stars[0].affine_map2(self.get_affine())]),
                    <[_]>::into_vec(box [parent_axis_aligned_input_bounds[0].clone()]),
                    false,
                )
            }
        }
        impl fmt::Display for Interpolate {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_fmt(::core::fmt::Arguments::new_v1(
                    &["Interpolate ", " -> ", " "],
                    &[
                        ::core::fmt::ArgumentV1::new_display(&self.input_shape),
                        ::core::fmt::ArgumentV1::new_display(&self.output_shape),
                        ::core::fmt::ArgumentV1::new_debug(&self.method),
                    ],
                ))
            }
        }
    }
    pub mod relu {
        use crate::affine::Affine2;
        use crate::bounds::Bounds1;
        use crate::graph::Operation;
        use crate::star::Star2;
        use crate::NNVFloat;
        use ndarray::Array1;
        use ndarray::Array2;
        use ndarray::ArrayView1;
        use ndarray::ArrayViewMut1;
        use ndarray::Zip;
        use num::Float;
        use num::Zero;
        use serde::{Deserialize, Serialize};
        use std::fmt::{Display, Formatter, Result};
        use std::ops::Deref;
        use std::ops::Neg;
        pub struct ReLU {
            ndims: usize,
        }
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl ::core::clone::Clone for ReLU {
            #[inline]
            fn clone(&self) -> ReLU {
                match *self {
                    ReLU {
                        ndims: ref __self_0_0,
                    } => ReLU {
                        ndims: ::core::clone::Clone::clone(&(*__self_0_0)),
                    },
                }
            }
        }
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl ::core::fmt::Debug for ReLU {
            fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                match *self {
                    ReLU {
                        ndims: ref __self_0_0,
                    } => {
                        let debug_trait_builder =
                            &mut ::core::fmt::Formatter::debug_struct(f, "ReLU");
                        let _ = ::core::fmt::DebugStruct::field(
                            debug_trait_builder,
                            "ndims",
                            &&(*__self_0_0),
                        );
                        ::core::fmt::DebugStruct::finish(debug_trait_builder)
                    }
                }
            }
        }
        #[doc(hidden)]
        #[allow(non_upper_case_globals, unused_attributes, unused_qualifications)]
        const _: () = {
            #[allow(unused_extern_crates, clippy::useless_attribute)]
            extern crate serde as _serde;
            #[automatically_derived]
            impl<'de> _serde::Deserialize<'de> for ReLU {
                fn deserialize<__D>(
                    __deserializer: __D,
                ) -> _serde::__private::Result<Self, __D::Error>
                where
                    __D: _serde::Deserializer<'de>,
                {
                    #[allow(non_camel_case_types)]
                    enum __Field {
                        __field0,
                        __ignore,
                    }
                    struct __FieldVisitor;
                    impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                        type Value = __Field;
                        fn expecting(
                            &self,
                            __formatter: &mut _serde::__private::Formatter,
                        ) -> _serde::__private::fmt::Result {
                            _serde::__private::Formatter::write_str(__formatter, "field identifier")
                        }
                        fn visit_u64<__E>(
                            self,
                            __value: u64,
                        ) -> _serde::__private::Result<Self::Value, __E>
                        where
                            __E: _serde::de::Error,
                        {
                            match __value {
                                0u64 => _serde::__private::Ok(__Field::__field0),
                                _ => _serde::__private::Ok(__Field::__ignore),
                            }
                        }
                        fn visit_str<__E>(
                            self,
                            __value: &str,
                        ) -> _serde::__private::Result<Self::Value, __E>
                        where
                            __E: _serde::de::Error,
                        {
                            match __value {
                                "ndims" => _serde::__private::Ok(__Field::__field0),
                                _ => _serde::__private::Ok(__Field::__ignore),
                            }
                        }
                        fn visit_bytes<__E>(
                            self,
                            __value: &[u8],
                        ) -> _serde::__private::Result<Self::Value, __E>
                        where
                            __E: _serde::de::Error,
                        {
                            match __value {
                                b"ndims" => _serde::__private::Ok(__Field::__field0),
                                _ => _serde::__private::Ok(__Field::__ignore),
                            }
                        }
                    }
                    impl<'de> _serde::Deserialize<'de> for __Field {
                        #[inline]
                        fn deserialize<__D>(
                            __deserializer: __D,
                        ) -> _serde::__private::Result<Self, __D::Error>
                        where
                            __D: _serde::Deserializer<'de>,
                        {
                            _serde::Deserializer::deserialize_identifier(
                                __deserializer,
                                __FieldVisitor,
                            )
                        }
                    }
                    struct __Visitor<'de> {
                        marker: _serde::__private::PhantomData<ReLU>,
                        lifetime: _serde::__private::PhantomData<&'de ()>,
                    }
                    impl<'de> _serde::de::Visitor<'de> for __Visitor<'de> {
                        type Value = ReLU;
                        fn expecting(
                            &self,
                            __formatter: &mut _serde::__private::Formatter,
                        ) -> _serde::__private::fmt::Result {
                            _serde::__private::Formatter::write_str(__formatter, "struct ReLU")
                        }
                        #[inline]
                        fn visit_seq<__A>(
                            self,
                            mut __seq: __A,
                        ) -> _serde::__private::Result<Self::Value, __A::Error>
                        where
                            __A: _serde::de::SeqAccess<'de>,
                        {
                            let __field0 = match match _serde::de::SeqAccess::next_element::<usize>(
                                &mut __seq,
                            ) {
                                _serde::__private::Ok(__val) => __val,
                                _serde::__private::Err(__err) => {
                                    return _serde::__private::Err(__err);
                                }
                            } {
                                _serde::__private::Some(__value) => __value,
                                _serde::__private::None => {
                                    return _serde::__private::Err(
                                        _serde::de::Error::invalid_length(
                                            0usize,
                                            &"struct ReLU with 1 element",
                                        ),
                                    );
                                }
                            };
                            _serde::__private::Ok(ReLU { ndims: __field0 })
                        }
                        #[inline]
                        fn visit_map<__A>(
                            self,
                            mut __map: __A,
                        ) -> _serde::__private::Result<Self::Value, __A::Error>
                        where
                            __A: _serde::de::MapAccess<'de>,
                        {
                            let mut __field0: _serde::__private::Option<usize> =
                                _serde::__private::None;
                            while let _serde::__private::Some(__key) =
                                match _serde::de::MapAccess::next_key::<__Field>(&mut __map) {
                                    _serde::__private::Ok(__val) => __val,
                                    _serde::__private::Err(__err) => {
                                        return _serde::__private::Err(__err);
                                    }
                                }
                            {
                                match __key {
                                    __Field::__field0 => {
                                        if _serde::__private::Option::is_some(&__field0) {
                                            return _serde::__private::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field(
                                                    "ndims",
                                                ),
                                            );
                                        }
                                        __field0 = _serde::__private::Some(
                                            match _serde::de::MapAccess::next_value::<usize>(
                                                &mut __map,
                                            ) {
                                                _serde::__private::Ok(__val) => __val,
                                                _serde::__private::Err(__err) => {
                                                    return _serde::__private::Err(__err);
                                                }
                                            },
                                        );
                                    }
                                    _ => {
                                        let _ = match _serde::de::MapAccess::next_value::<
                                            _serde::de::IgnoredAny,
                                        >(
                                            &mut __map
                                        ) {
                                            _serde::__private::Ok(__val) => __val,
                                            _serde::__private::Err(__err) => {
                                                return _serde::__private::Err(__err);
                                            }
                                        };
                                    }
                                }
                            }
                            let __field0 = match __field0 {
                                _serde::__private::Some(__field0) => __field0,
                                _serde::__private::None => {
                                    match _serde::__private::de::missing_field("ndims") {
                                        _serde::__private::Ok(__val) => __val,
                                        _serde::__private::Err(__err) => {
                                            return _serde::__private::Err(__err);
                                        }
                                    }
                                }
                            };
                            _serde::__private::Ok(ReLU { ndims: __field0 })
                        }
                    }
                    const FIELDS: &'static [&'static str] = &["ndims"];
                    _serde::Deserializer::deserialize_struct(
                        __deserializer,
                        "ReLU",
                        FIELDS,
                        __Visitor {
                            marker: _serde::__private::PhantomData::<ReLU>,
                            lifetime: _serde::__private::PhantomData,
                        },
                    )
                }
            }
        };
        #[doc(hidden)]
        #[allow(non_upper_case_globals, unused_attributes, unused_qualifications)]
        const _: () = {
            #[allow(unused_extern_crates, clippy::useless_attribute)]
            extern crate serde as _serde;
            #[automatically_derived]
            impl _serde::Serialize for ReLU {
                fn serialize<__S>(
                    &self,
                    __serializer: __S,
                ) -> _serde::__private::Result<__S::Ok, __S::Error>
                where
                    __S: _serde::Serializer,
                {
                    let mut __serde_state = match _serde::Serializer::serialize_struct(
                        __serializer,
                        "ReLU",
                        false as usize + 1,
                    ) {
                        _serde::__private::Ok(__val) => __val,
                        _serde::__private::Err(__err) => {
                            return _serde::__private::Err(__err);
                        }
                    };
                    match _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "ndims",
                        &self.ndims,
                    ) {
                        _serde::__private::Ok(__val) => __val,
                        _serde::__private::Err(__err) => {
                            return _serde::__private::Err(__err);
                        }
                    };
                    _serde::ser::SerializeStruct::end(__serde_state)
                }
            }
        };
        impl ::core::marker::StructuralPartialEq for ReLU {}
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl ::core::cmp::PartialEq for ReLU {
            #[inline]
            fn eq(&self, other: &ReLU) -> bool {
                match *other {
                    ReLU {
                        ndims: ref __self_1_0,
                    } => match *self {
                        ReLU {
                            ndims: ref __self_0_0,
                        } => (*__self_0_0) == (*__self_1_0),
                    },
                }
            }
            #[inline]
            fn ne(&self, other: &ReLU) -> bool {
                match *other {
                    ReLU {
                        ndims: ref __self_1_0,
                    } => match *self {
                        ReLU {
                            ndims: ref __self_0_0,
                        } => (*__self_0_0) != (*__self_1_0),
                    },
                }
            }
        }
        impl ReLU {
            pub const fn new(ndims: usize) -> Self {
                Self { ndims }
            }
        }
        impl Display for ReLU {
            fn fmt(&self, f: &mut Formatter) -> Result {
                f.write_fmt(::core::fmt::Arguments::new_v1(&["ReLU"], &[]))
            }
        }
        impl Operation for ReLU {
            fn as_any(&self) -> &dyn std::any::Any {
                self
            }
            fn num_steps(&self) -> Option<usize> {
                Some(self.ndims)
            }
            fn inputs_dims(&self) -> Vec<usize> {
                <[_]>::into_vec(box [self.ndims])
            }
            fn outputs_dims(&self) -> Vec<usize> {
                <[_]>::into_vec(box [self.ndims])
            }
            fn forward1(&self, input: &[&Array1<NNVFloat>]) -> Vec<Array1<NNVFloat>> {
                <[_]>::into_vec(box [input[0].mapv(|x| if x.lt(&0.) { 0. } else { x })])
            }
            fn forward2(&self, input: &[&Array2<NNVFloat>]) -> Vec<Array2<NNVFloat>> {
                <[_]>::into_vec(box [input[0].mapv(|x| if x.lt(&0.) { 0. } else { x })])
            }
            fn apply_bounds(
                &self,
                bounds: &[Bounds1],
                lower_aff: &[Affine2],
                upper_aff: &[Affine2],
            ) -> Vec<(Bounds1, Affine2, Affine2)> {
                if (self.ndims + 1) == bounds[0].ndim() {
                    <[_]>::into_vec(box [deep_poly_relu(&bounds[0], &lower_aff[0], &upper_aff[0])])
                } else {
                    let (bounds_head, bounds_tail) = bounds[0].split_at(self.ndims);
                    let (lower_aff_head, lower_aff_tail) = lower_aff[0].split_at(self.ndims);
                    let (upper_aff_head, upper_aff_tail) = lower_aff[0].split_at(self.ndims);
                    let (bounds_part, lower_part, upper_part) =
                        deep_poly_relu(&bounds_head, &lower_aff_head, &upper_aff_head);
                    <[_]>::into_vec(box [(
                        bounds_part.append(&bounds_tail),
                        lower_part.append(&lower_aff_tail),
                        upper_part.append(&upper_aff_tail),
                    )])
                }
            }
            fn apply_bounds_step(
                &self,
                dim: usize,
                bounds: &[Bounds1],
                lower_aff: &[Affine2],
                upper_aff: &[Affine2],
            ) -> Vec<(Bounds1, Affine2, Affine2)> {
                <[_]>::into_vec(box [deep_poly_steprelu(
                    dim,
                    bounds[0].clone(),
                    lower_aff[0].clone(),
                    upper_aff[0].clone(),
                )])
            }
            fn get_activation_pattern(
                &self,
                state: &[&Array2<NNVFloat>],
            ) -> Option<Vec<Array2<bool>>> {
                Some(<[_]>::into_vec(box [state[0].mapv(|x| x >= 0.0)]))
            }
            fn forward_star<StarRef: Deref<Target = Star2>>(
                &self,
                stars: Vec<StarRef>,
                dim: Option<usize>,
                parent_axis_aligned_input_bounds: Vec<&Bounds1>,
            ) -> (Vec<Star2>, Vec<Bounds1>, bool) {
                match (&1, &parent_axis_aligned_input_bounds.len()) {
                    (left_val, right_val) => {
                        if !(*left_val == *right_val) {
                            let kind = ::core::panicking::AssertKind::Eq;
                            ::core::panicking::assert_failed(
                                kind,
                                &*left_val,
                                &*right_val,
                                ::core::option::Option::None,
                            );
                        }
                    }
                };
                let parent_aa_input_bounds = parent_axis_aligned_input_bounds[0];
                match (&1, &stars.len()) {
                    (left_val, right_val) => {
                        if !(*left_val == *right_val) {
                            let kind = ::core::panicking::AssertKind::Eq;
                            ::core::panicking::assert_failed(
                                kind,
                                &*left_val,
                                &*right_val,
                                ::core::option::Option::None,
                            );
                        }
                    }
                };
                let star = stars[0];
                let dim = dim.unwrap();
                let child_stars = star.step_relu2(dim, Some(parent_aa_input_bounds));
                let mut same_output_bounds = false;
                let mut stars = ::alloc::vec::Vec::new();
                let mut star_input_bounds = ::alloc::vec::Vec::new();
                let is_single_child = child_stars.0.is_some() ^ child_stars.1.is_some();
                if let Some(mut lower_star) = child_stars.0 {
                    let mut bounds = parent_aa_input_bounds.clone();
                    bounds.index_mut(dim)[0] = 0.;
                    bounds.index_mut(dim)[1] = 0.;
                    if is_single_child {
                        let num_constraints = lower_star.num_constraints();
                        lower_star = lower_star.remove_constraint(num_constraints - 1);
                    }
                    stars.push(lower_star);
                    star_input_bounds.push(bounds);
                }
                if let Some(mut upper_star) = child_stars.1 {
                    let mut bounds = parent_aa_input_bounds.clone();
                    let mut lb = bounds.index_mut(dim);
                    if lb[0].is_sign_negative() {
                        lb[0] = 0.;
                    }
                    if is_single_child {
                        let num_constraints = upper_star.num_constraints();
                        upper_star = upper_star.remove_constraint(num_constraints - 1);
                        same_output_bounds = true;
                    }
                    stars.push(upper_star);
                    star_input_bounds.push(bounds);
                }
                (stars, star_input_bounds, same_output_bounds)
            }
        }
        /// # Panics
        pub fn deep_poly_steprelu(
            dim: usize,
            mut bounds: Bounds1,
            mut lower_aff: Affine2,
            mut upper_aff: Affine2,
        ) -> (Bounds1, Affine2, Affine2) {
            let mut bounds_slice = bounds.index_mut(dim);
            let (mut lbasis, mut lshift) = lower_aff.get_eqn_mut(dim);
            let (mut u_basis, mut u_shift) = upper_aff.get_eqn_mut(dim);
            let l = bounds_slice[[0]];
            let u = bounds_slice[[1]];
            if u <= NNVFloat::zero() {
                bounds_slice.fill(NNVFloat::zero());
                lbasis.fill(NNVFloat::zero());
                u_basis.fill(NNVFloat::zero());
                lshift.fill(NNVFloat::zero());
                u_shift.fill(NNVFloat::zero());
            } else if l >= NNVFloat::zero() {
            } else {
                if u == NNVFloat::infinity() {
                    u_basis.mapv_inplace(|x| {
                        if x * NNVFloat::infinity() == NNVFloat::nan() {
                            0.
                        } else {
                            NNVFloat::INFINITY
                        }
                    });
                    u_shift.mapv_inplace(|x| {
                        if x * NNVFloat::infinity() == NNVFloat::nan() {
                            0.
                        } else {
                            NNVFloat::INFINITY
                        }
                    });
                } else {
                    u_basis.mapv_inplace(|a| a * (u / (u - l)));
                    u_shift.mapv_inplace(|b| u * (b - l) / (u - l));
                }
                if u < NNVFloat::neg(l) || l == NNVFloat::neg_infinity() {
                    *bounds_slice.get_mut(0).unwrap() = NNVFloat::zero();
                    lbasis.fill(NNVFloat::zero());
                    lshift.fill(NNVFloat::zero());
                } else {
                }
            }
            (bounds, lower_aff, upper_aff)
        }
        pub fn deep_poly_relu(
            bounds: &Bounds1,
            lower_aff: &Affine2,
            upper_aff: &Affine2,
        ) -> (Bounds1, Affine2, Affine2) {
            let mut out = bounds.clone();
            let mut l_mul = Array1::ones(bounds.ndim());
            let mut u_mul = Array1::ones(bounds.ndim());
            let mut u_shift = Array1::zeros(bounds.ndim());
            Zip::from(bounds.bounds_iter())
                .and(out.bounds_iter_mut())
                .and(&mut l_mul)
                .and(&mut u_mul)
                .and(&mut u_shift)
                .for_each(
                    |b: ArrayView1<NNVFloat>,
                     mut out: ArrayViewMut1<NNVFloat>,
                     l_mul: &mut NNVFloat,
                     u_mul: &mut NNVFloat,
                     u_shift: &mut NNVFloat| {
                        let l = b[0];
                        let u = b[1];
                        if u <= NNVFloat::zero() {
                            out[0] = NNVFloat::zero();
                            out[1] = NNVFloat::zero();
                            *l_mul = NNVFloat::zero();
                            *u_mul = NNVFloat::zero();
                        } else if l >= NNVFloat::zero() {
                        } else {
                            *u_mul = u / (u - l);
                            *u_shift = NNVFloat::neg((u * l) / (u - l));
                            if u < NNVFloat::neg(l) {
                                out[0] = NNVFloat::zero();
                                *l_mul = NNVFloat::zero();
                            } else {
                            }
                        }
                    },
                );
            let mut lower_aff = lower_aff.clone();
            lower_aff.scale_eqns(l_mul.view());
            let mut upper_aff = upper_aff.clone();
            upper_aff.scale_eqns(u_mul.view());
            upper_aff = upper_aff + u_shift;
            (out, lower_aff, upper_aff)
        }
    }
    pub use conv::Conv;
    pub use dense::Dense;
    pub use dnn::DNN;
    pub use interpolate::Interpolate;
    pub use relu::ReLU;
}
pub mod gaussian {
    use crate::NNVFloat;
    use ndarray::Array1;
    use ndarray::Array2;
    use ndarray::Ix2;
    use ndarray_rand::rand_distr::StandardNormal;
    use ndarray_rand::RandomExt;
    use num::One;
    use rand::Rng;
    use serde::{Deserialize, Serialize};
    use truncnorm::distributions::MultivariateTruncatedNormal;
    use truncnorm::tilting::TiltingSolution;
    pub enum GaussianDistribution {
        Gaussian {
            loc: Array1<NNVFloat>,
            scale: Array1<NNVFloat>,
        },
        TruncGaussian {
            distribution: MultivariateTruncatedNormal<Ix2>,
            inv_coeffs: Array2<NNVFloat>,
        },
    }
    #[automatically_derived]
    #[allow(unused_qualifications)]
    impl ::core::fmt::Debug for GaussianDistribution {
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            match (&*self,) {
                (&GaussianDistribution::Gaussian {
                    loc: ref __self_0,
                    scale: ref __self_1,
                },) => {
                    let debug_trait_builder =
                        &mut ::core::fmt::Formatter::debug_struct(f, "Gaussian");
                    let _ =
                        ::core::fmt::DebugStruct::field(debug_trait_builder, "loc", &&(*__self_0));
                    let _ = ::core::fmt::DebugStruct::field(
                        debug_trait_builder,
                        "scale",
                        &&(*__self_1),
                    );
                    ::core::fmt::DebugStruct::finish(debug_trait_builder)
                }
                (&GaussianDistribution::TruncGaussian {
                    distribution: ref __self_0,
                    inv_coeffs: ref __self_1,
                },) => {
                    let debug_trait_builder =
                        &mut ::core::fmt::Formatter::debug_struct(f, "TruncGaussian");
                    let _ = ::core::fmt::DebugStruct::field(
                        debug_trait_builder,
                        "distribution",
                        &&(*__self_0),
                    );
                    let _ = ::core::fmt::DebugStruct::field(
                        debug_trait_builder,
                        "inv_coeffs",
                        &&(*__self_1),
                    );
                    ::core::fmt::DebugStruct::finish(debug_trait_builder)
                }
            }
        }
    }
    #[automatically_derived]
    #[allow(unused_qualifications)]
    impl ::core::clone::Clone for GaussianDistribution {
        #[inline]
        fn clone(&self) -> GaussianDistribution {
            match (&*self,) {
                (&GaussianDistribution::Gaussian {
                    loc: ref __self_0,
                    scale: ref __self_1,
                },) => GaussianDistribution::Gaussian {
                    loc: ::core::clone::Clone::clone(&(*__self_0)),
                    scale: ::core::clone::Clone::clone(&(*__self_1)),
                },
                (&GaussianDistribution::TruncGaussian {
                    distribution: ref __self_0,
                    inv_coeffs: ref __self_1,
                },) => GaussianDistribution::TruncGaussian {
                    distribution: ::core::clone::Clone::clone(&(*__self_0)),
                    inv_coeffs: ::core::clone::Clone::clone(&(*__self_1)),
                },
            }
        }
    }
    #[doc(hidden)]
    #[allow(non_upper_case_globals, unused_attributes, unused_qualifications)]
    const _: () = {
        #[allow(unused_extern_crates, clippy::useless_attribute)]
        extern crate serde as _serde;
        #[automatically_derived]
        impl<'de> _serde::Deserialize<'de> for GaussianDistribution {
            fn deserialize<__D>(__deserializer: __D) -> _serde::__private::Result<Self, __D::Error>
            where
                __D: _serde::Deserializer<'de>,
            {
                #[allow(non_camel_case_types)]
                enum __Field {
                    __field0,
                    __field1,
                }
                struct __FieldVisitor;
                impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                    type Value = __Field;
                    fn expecting(
                        &self,
                        __formatter: &mut _serde::__private::Formatter,
                    ) -> _serde::__private::fmt::Result {
                        _serde::__private::Formatter::write_str(__formatter, "variant identifier")
                    }
                    fn visit_u64<__E>(
                        self,
                        __value: u64,
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            0u64 => _serde::__private::Ok(__Field::__field0),
                            1u64 => _serde::__private::Ok(__Field::__field1),
                            _ => _serde::__private::Err(_serde::de::Error::invalid_value(
                                _serde::de::Unexpected::Unsigned(__value),
                                &"variant index 0 <= i < 2",
                            )),
                        }
                    }
                    fn visit_str<__E>(
                        self,
                        __value: &str,
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            "Gaussian" => _serde::__private::Ok(__Field::__field0),
                            "TruncGaussian" => _serde::__private::Ok(__Field::__field1),
                            _ => _serde::__private::Err(_serde::de::Error::unknown_variant(
                                __value, VARIANTS,
                            )),
                        }
                    }
                    fn visit_bytes<__E>(
                        self,
                        __value: &[u8],
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            b"Gaussian" => _serde::__private::Ok(__Field::__field0),
                            b"TruncGaussian" => _serde::__private::Ok(__Field::__field1),
                            _ => {
                                let __value = &_serde::__private::from_utf8_lossy(__value);
                                _serde::__private::Err(_serde::de::Error::unknown_variant(
                                    __value, VARIANTS,
                                ))
                            }
                        }
                    }
                }
                impl<'de> _serde::Deserialize<'de> for __Field {
                    #[inline]
                    fn deserialize<__D>(
                        __deserializer: __D,
                    ) -> _serde::__private::Result<Self, __D::Error>
                    where
                        __D: _serde::Deserializer<'de>,
                    {
                        _serde::Deserializer::deserialize_identifier(__deserializer, __FieldVisitor)
                    }
                }
                struct __Visitor<'de> {
                    marker: _serde::__private::PhantomData<GaussianDistribution>,
                    lifetime: _serde::__private::PhantomData<&'de ()>,
                }
                impl<'de> _serde::de::Visitor<'de> for __Visitor<'de> {
                    type Value = GaussianDistribution;
                    fn expecting(
                        &self,
                        __formatter: &mut _serde::__private::Formatter,
                    ) -> _serde::__private::fmt::Result {
                        _serde::__private::Formatter::write_str(
                            __formatter,
                            "enum GaussianDistribution",
                        )
                    }
                    fn visit_enum<__A>(
                        self,
                        __data: __A,
                    ) -> _serde::__private::Result<Self::Value, __A::Error>
                    where
                        __A: _serde::de::EnumAccess<'de>,
                    {
                        match match _serde::de::EnumAccess::variant(__data) {
                            _serde::__private::Ok(__val) => __val,
                            _serde::__private::Err(__err) => {
                                return _serde::__private::Err(__err);
                            }
                        } {
                            (__Field::__field0, __variant) => {
                                #[allow(non_camel_case_types)]
                                enum __Field {
                                    __field0,
                                    __field1,
                                    __ignore,
                                }
                                struct __FieldVisitor;
                                impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                                    type Value = __Field;
                                    fn expecting(
                                        &self,
                                        __formatter: &mut _serde::__private::Formatter,
                                    ) -> _serde::__private::fmt::Result
                                    {
                                        _serde::__private::Formatter::write_str(
                                            __formatter,
                                            "field identifier",
                                        )
                                    }
                                    fn visit_u64<__E>(
                                        self,
                                        __value: u64,
                                    ) -> _serde::__private::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            0u64 => _serde::__private::Ok(__Field::__field0),
                                            1u64 => _serde::__private::Ok(__Field::__field1),
                                            _ => _serde::__private::Ok(__Field::__ignore),
                                        }
                                    }
                                    fn visit_str<__E>(
                                        self,
                                        __value: &str,
                                    ) -> _serde::__private::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            "loc" => _serde::__private::Ok(__Field::__field0),
                                            "scale" => _serde::__private::Ok(__Field::__field1),
                                            _ => _serde::__private::Ok(__Field::__ignore),
                                        }
                                    }
                                    fn visit_bytes<__E>(
                                        self,
                                        __value: &[u8],
                                    ) -> _serde::__private::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            b"loc" => _serde::__private::Ok(__Field::__field0),
                                            b"scale" => _serde::__private::Ok(__Field::__field1),
                                            _ => _serde::__private::Ok(__Field::__ignore),
                                        }
                                    }
                                }
                                impl<'de> _serde::Deserialize<'de> for __Field {
                                    #[inline]
                                    fn deserialize<__D>(
                                        __deserializer: __D,
                                    ) -> _serde::__private::Result<Self, __D::Error>
                                    where
                                        __D: _serde::Deserializer<'de>,
                                    {
                                        _serde::Deserializer::deserialize_identifier(
                                            __deserializer,
                                            __FieldVisitor,
                                        )
                                    }
                                }
                                struct __Visitor<'de> {
                                    marker: _serde::__private::PhantomData<GaussianDistribution>,
                                    lifetime: _serde::__private::PhantomData<&'de ()>,
                                }
                                impl<'de> _serde::de::Visitor<'de> for __Visitor<'de> {
                                    type Value = GaussianDistribution;
                                    fn expecting(
                                        &self,
                                        __formatter: &mut _serde::__private::Formatter,
                                    ) -> _serde::__private::fmt::Result
                                    {
                                        _serde::__private::Formatter::write_str(
                                            __formatter,
                                            "struct variant GaussianDistribution::Gaussian",
                                        )
                                    }
                                    #[inline]
                                    fn visit_seq<__A>(
                                        self,
                                        mut __seq: __A,
                                    ) -> _serde::__private::Result<Self::Value, __A::Error>
                                    where
                                        __A: _serde::de::SeqAccess<'de>,
                                    {
                                        let __field0 =
                                            match match _serde::de::SeqAccess::next_element::<
                                                Array1<NNVFloat>,
                                            >(
                                                &mut __seq
                                            ) {
                                                _serde::__private::Ok(__val) => __val,
                                                _serde::__private::Err(__err) => {
                                                    return _serde::__private::Err(__err);
                                                }
                                            } {
                                                _serde::__private::Some(__value) => __value,
                                                _serde::__private::None => {
                                                    return _serde :: __private :: Err (_serde :: de :: Error :: invalid_length (0usize , & "struct variant GaussianDistribution::Gaussian with 2 elements")) ;
                                                }
                                            };
                                        let __field1 =
                                            match match _serde::de::SeqAccess::next_element::<
                                                Array1<NNVFloat>,
                                            >(
                                                &mut __seq
                                            ) {
                                                _serde::__private::Ok(__val) => __val,
                                                _serde::__private::Err(__err) => {
                                                    return _serde::__private::Err(__err);
                                                }
                                            } {
                                                _serde::__private::Some(__value) => __value,
                                                _serde::__private::None => {
                                                    return _serde :: __private :: Err (_serde :: de :: Error :: invalid_length (1usize , & "struct variant GaussianDistribution::Gaussian with 2 elements")) ;
                                                }
                                            };
                                        _serde::__private::Ok(GaussianDistribution::Gaussian {
                                            loc: __field0,
                                            scale: __field1,
                                        })
                                    }
                                    #[inline]
                                    fn visit_map<__A>(
                                        self,
                                        mut __map: __A,
                                    ) -> _serde::__private::Result<Self::Value, __A::Error>
                                    where
                                        __A: _serde::de::MapAccess<'de>,
                                    {
                                        let mut __field0: _serde::__private::Option<
                                            Array1<NNVFloat>,
                                        > = _serde::__private::None;
                                        let mut __field1: _serde::__private::Option<
                                            Array1<NNVFloat>,
                                        > = _serde::__private::None;
                                        while let _serde::__private::Some(__key) =
                                            match _serde::de::MapAccess::next_key::<__Field>(
                                                &mut __map,
                                            ) {
                                                _serde::__private::Ok(__val) => __val,
                                                _serde::__private::Err(__err) => {
                                                    return _serde::__private::Err(__err);
                                                }
                                            }
                                        {
                                            match __key {
                                                __Field::__field0 => {
                                                    if _serde::__private::Option::is_some(&__field0)
                                                    {
                                                        return _serde :: __private :: Err (< __A :: Error as _serde :: de :: Error > :: duplicate_field ("loc")) ;
                                                    }
                                                    __field0 = _serde::__private::Some(
                                                        match _serde::de::MapAccess::next_value::<
                                                            Array1<NNVFloat>,
                                                        >(
                                                            &mut __map
                                                        ) {
                                                            _serde::__private::Ok(__val) => __val,
                                                            _serde::__private::Err(__err) => {
                                                                return _serde::__private::Err(
                                                                    __err,
                                                                );
                                                            }
                                                        },
                                                    );
                                                }
                                                __Field::__field1 => {
                                                    if _serde::__private::Option::is_some(&__field1)
                                                    {
                                                        return _serde :: __private :: Err (< __A :: Error as _serde :: de :: Error > :: duplicate_field ("scale")) ;
                                                    }
                                                    __field1 = _serde::__private::Some(
                                                        match _serde::de::MapAccess::next_value::<
                                                            Array1<NNVFloat>,
                                                        >(
                                                            &mut __map
                                                        ) {
                                                            _serde::__private::Ok(__val) => __val,
                                                            _serde::__private::Err(__err) => {
                                                                return _serde::__private::Err(
                                                                    __err,
                                                                );
                                                            }
                                                        },
                                                    );
                                                }
                                                _ => {
                                                    let _ = match _serde::de::MapAccess::next_value::<
                                                        _serde::de::IgnoredAny,
                                                    >(
                                                        &mut __map
                                                    ) {
                                                        _serde::__private::Ok(__val) => __val,
                                                        _serde::__private::Err(__err) => {
                                                            return _serde::__private::Err(__err);
                                                        }
                                                    };
                                                }
                                            }
                                        }
                                        let __field0 = match __field0 {
                                            _serde::__private::Some(__field0) => __field0,
                                            _serde::__private::None => {
                                                match _serde::__private::de::missing_field("loc") {
                                                    _serde::__private::Ok(__val) => __val,
                                                    _serde::__private::Err(__err) => {
                                                        return _serde::__private::Err(__err);
                                                    }
                                                }
                                            }
                                        };
                                        let __field1 = match __field1 {
                                            _serde::__private::Some(__field1) => __field1,
                                            _serde::__private::None => {
                                                match _serde::__private::de::missing_field("scale")
                                                {
                                                    _serde::__private::Ok(__val) => __val,
                                                    _serde::__private::Err(__err) => {
                                                        return _serde::__private::Err(__err);
                                                    }
                                                }
                                            }
                                        };
                                        _serde::__private::Ok(GaussianDistribution::Gaussian {
                                            loc: __field0,
                                            scale: __field1,
                                        })
                                    }
                                }
                                const FIELDS: &'static [&'static str] = &["loc", "scale"];
                                _serde::de::VariantAccess::struct_variant(
                                    __variant,
                                    FIELDS,
                                    __Visitor {
                                        marker: _serde::__private::PhantomData::<
                                            GaussianDistribution,
                                        >,
                                        lifetime: _serde::__private::PhantomData,
                                    },
                                )
                            }
                            (__Field::__field1, __variant) => {
                                #[allow(non_camel_case_types)]
                                enum __Field {
                                    __field0,
                                    __field1,
                                    __ignore,
                                }
                                struct __FieldVisitor;
                                impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                                    type Value = __Field;
                                    fn expecting(
                                        &self,
                                        __formatter: &mut _serde::__private::Formatter,
                                    ) -> _serde::__private::fmt::Result
                                    {
                                        _serde::__private::Formatter::write_str(
                                            __formatter,
                                            "field identifier",
                                        )
                                    }
                                    fn visit_u64<__E>(
                                        self,
                                        __value: u64,
                                    ) -> _serde::__private::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            0u64 => _serde::__private::Ok(__Field::__field0),
                                            1u64 => _serde::__private::Ok(__Field::__field1),
                                            _ => _serde::__private::Ok(__Field::__ignore),
                                        }
                                    }
                                    fn visit_str<__E>(
                                        self,
                                        __value: &str,
                                    ) -> _serde::__private::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            "distribution" => {
                                                _serde::__private::Ok(__Field::__field0)
                                            }
                                            "inv_coeffs" => {
                                                _serde::__private::Ok(__Field::__field1)
                                            }
                                            _ => _serde::__private::Ok(__Field::__ignore),
                                        }
                                    }
                                    fn visit_bytes<__E>(
                                        self,
                                        __value: &[u8],
                                    ) -> _serde::__private::Result<Self::Value, __E>
                                    where
                                        __E: _serde::de::Error,
                                    {
                                        match __value {
                                            b"distribution" => {
                                                _serde::__private::Ok(__Field::__field0)
                                            }
                                            b"inv_coeffs" => {
                                                _serde::__private::Ok(__Field::__field1)
                                            }
                                            _ => _serde::__private::Ok(__Field::__ignore),
                                        }
                                    }
                                }
                                impl<'de> _serde::Deserialize<'de> for __Field {
                                    #[inline]
                                    fn deserialize<__D>(
                                        __deserializer: __D,
                                    ) -> _serde::__private::Result<Self, __D::Error>
                                    where
                                        __D: _serde::Deserializer<'de>,
                                    {
                                        _serde::Deserializer::deserialize_identifier(
                                            __deserializer,
                                            __FieldVisitor,
                                        )
                                    }
                                }
                                struct __Visitor<'de> {
                                    marker: _serde::__private::PhantomData<GaussianDistribution>,
                                    lifetime: _serde::__private::PhantomData<&'de ()>,
                                }
                                impl<'de> _serde::de::Visitor<'de> for __Visitor<'de> {
                                    type Value = GaussianDistribution;
                                    fn expecting(
                                        &self,
                                        __formatter: &mut _serde::__private::Formatter,
                                    ) -> _serde::__private::fmt::Result
                                    {
                                        _serde::__private::Formatter::write_str(
                                            __formatter,
                                            "struct variant GaussianDistribution::TruncGaussian",
                                        )
                                    }
                                    #[inline]
                                    fn visit_seq<__A>(
                                        self,
                                        mut __seq: __A,
                                    ) -> _serde::__private::Result<Self::Value, __A::Error>
                                    where
                                        __A: _serde::de::SeqAccess<'de>,
                                    {
                                        let __field0 =
                                            match match _serde::de::SeqAccess::next_element::<
                                                MultivariateTruncatedNormal<Ix2>,
                                            >(
                                                &mut __seq
                                            ) {
                                                _serde::__private::Ok(__val) => __val,
                                                _serde::__private::Err(__err) => {
                                                    return _serde::__private::Err(__err);
                                                }
                                            } {
                                                _serde::__private::Some(__value) => __value,
                                                _serde::__private::None => {
                                                    return _serde :: __private :: Err (_serde :: de :: Error :: invalid_length (0usize , & "struct variant GaussianDistribution::TruncGaussian with 2 elements")) ;
                                                }
                                            };
                                        let __field1 =
                                            match match _serde::de::SeqAccess::next_element::<
                                                Array2<NNVFloat>,
                                            >(
                                                &mut __seq
                                            ) {
                                                _serde::__private::Ok(__val) => __val,
                                                _serde::__private::Err(__err) => {
                                                    return _serde::__private::Err(__err);
                                                }
                                            } {
                                                _serde::__private::Some(__value) => __value,
                                                _serde::__private::None => {
                                                    return _serde :: __private :: Err (_serde :: de :: Error :: invalid_length (1usize , & "struct variant GaussianDistribution::TruncGaussian with 2 elements")) ;
                                                }
                                            };
                                        _serde::__private::Ok(GaussianDistribution::TruncGaussian {
                                            distribution: __field0,
                                            inv_coeffs: __field1,
                                        })
                                    }
                                    #[inline]
                                    fn visit_map<__A>(
                                        self,
                                        mut __map: __A,
                                    ) -> _serde::__private::Result<Self::Value, __A::Error>
                                    where
                                        __A: _serde::de::MapAccess<'de>,
                                    {
                                        let mut __field0: _serde::__private::Option<
                                            MultivariateTruncatedNormal<Ix2>,
                                        > = _serde::__private::None;
                                        let mut __field1: _serde::__private::Option<
                                            Array2<NNVFloat>,
                                        > = _serde::__private::None;
                                        while let _serde::__private::Some(__key) =
                                            match _serde::de::MapAccess::next_key::<__Field>(
                                                &mut __map,
                                            ) {
                                                _serde::__private::Ok(__val) => __val,
                                                _serde::__private::Err(__err) => {
                                                    return _serde::__private::Err(__err);
                                                }
                                            }
                                        {
                                            match __key {
                                                __Field::__field0 => {
                                                    if _serde::__private::Option::is_some(&__field0)
                                                    {
                                                        return _serde :: __private :: Err (< __A :: Error as _serde :: de :: Error > :: duplicate_field ("distribution")) ;
                                                    }
                                                    __field0 = _serde::__private::Some(
                                                        match _serde::de::MapAccess::next_value::<
                                                            MultivariateTruncatedNormal<Ix2>,
                                                        >(
                                                            &mut __map
                                                        ) {
                                                            _serde::__private::Ok(__val) => __val,
                                                            _serde::__private::Err(__err) => {
                                                                return _serde::__private::Err(
                                                                    __err,
                                                                );
                                                            }
                                                        },
                                                    );
                                                }
                                                __Field::__field1 => {
                                                    if _serde::__private::Option::is_some(&__field1)
                                                    {
                                                        return _serde :: __private :: Err (< __A :: Error as _serde :: de :: Error > :: duplicate_field ("inv_coeffs")) ;
                                                    }
                                                    __field1 = _serde::__private::Some(
                                                        match _serde::de::MapAccess::next_value::<
                                                            Array2<NNVFloat>,
                                                        >(
                                                            &mut __map
                                                        ) {
                                                            _serde::__private::Ok(__val) => __val,
                                                            _serde::__private::Err(__err) => {
                                                                return _serde::__private::Err(
                                                                    __err,
                                                                );
                                                            }
                                                        },
                                                    );
                                                }
                                                _ => {
                                                    let _ = match _serde::de::MapAccess::next_value::<
                                                        _serde::de::IgnoredAny,
                                                    >(
                                                        &mut __map
                                                    ) {
                                                        _serde::__private::Ok(__val) => __val,
                                                        _serde::__private::Err(__err) => {
                                                            return _serde::__private::Err(__err);
                                                        }
                                                    };
                                                }
                                            }
                                        }
                                        let __field0 = match __field0 {
                                            _serde::__private::Some(__field0) => __field0,
                                            _serde::__private::None => {
                                                match _serde::__private::de::missing_field(
                                                    "distribution",
                                                ) {
                                                    _serde::__private::Ok(__val) => __val,
                                                    _serde::__private::Err(__err) => {
                                                        return _serde::__private::Err(__err);
                                                    }
                                                }
                                            }
                                        };
                                        let __field1 = match __field1 {
                                            _serde::__private::Some(__field1) => __field1,
                                            _serde::__private::None => {
                                                match _serde::__private::de::missing_field(
                                                    "inv_coeffs",
                                                ) {
                                                    _serde::__private::Ok(__val) => __val,
                                                    _serde::__private::Err(__err) => {
                                                        return _serde::__private::Err(__err);
                                                    }
                                                }
                                            }
                                        };
                                        _serde::__private::Ok(GaussianDistribution::TruncGaussian {
                                            distribution: __field0,
                                            inv_coeffs: __field1,
                                        })
                                    }
                                }
                                const FIELDS: &'static [&'static str] =
                                    &["distribution", "inv_coeffs"];
                                _serde::de::VariantAccess::struct_variant(
                                    __variant,
                                    FIELDS,
                                    __Visitor {
                                        marker: _serde::__private::PhantomData::<
                                            GaussianDistribution,
                                        >,
                                        lifetime: _serde::__private::PhantomData,
                                    },
                                )
                            }
                        }
                    }
                }
                const VARIANTS: &'static [&'static str] = &["Gaussian", "TruncGaussian"];
                _serde::Deserializer::deserialize_enum(
                    __deserializer,
                    "GaussianDistribution",
                    VARIANTS,
                    __Visitor {
                        marker: _serde::__private::PhantomData::<GaussianDistribution>,
                        lifetime: _serde::__private::PhantomData,
                    },
                )
            }
        }
    };
    #[doc(hidden)]
    #[allow(non_upper_case_globals, unused_attributes, unused_qualifications)]
    const _: () = {
        #[allow(unused_extern_crates, clippy::useless_attribute)]
        extern crate serde as _serde;
        #[automatically_derived]
        impl _serde::Serialize for GaussianDistribution {
            fn serialize<__S>(
                &self,
                __serializer: __S,
            ) -> _serde::__private::Result<__S::Ok, __S::Error>
            where
                __S: _serde::Serializer,
            {
                match *self {
                    GaussianDistribution::Gaussian { ref loc, ref scale } => {
                        let mut __serde_state = match _serde::Serializer::serialize_struct_variant(
                            __serializer,
                            "GaussianDistribution",
                            0u32,
                            "Gaussian",
                            0 + 1 + 1,
                        ) {
                            _serde::__private::Ok(__val) => __val,
                            _serde::__private::Err(__err) => {
                                return _serde::__private::Err(__err);
                            }
                        };
                        match _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "loc",
                            loc,
                        ) {
                            _serde::__private::Ok(__val) => __val,
                            _serde::__private::Err(__err) => {
                                return _serde::__private::Err(__err);
                            }
                        };
                        match _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "scale",
                            scale,
                        ) {
                            _serde::__private::Ok(__val) => __val,
                            _serde::__private::Err(__err) => {
                                return _serde::__private::Err(__err);
                            }
                        };
                        _serde::ser::SerializeStructVariant::end(__serde_state)
                    }
                    GaussianDistribution::TruncGaussian {
                        ref distribution,
                        ref inv_coeffs,
                    } => {
                        let mut __serde_state = match _serde::Serializer::serialize_struct_variant(
                            __serializer,
                            "GaussianDistribution",
                            1u32,
                            "TruncGaussian",
                            0 + 1 + 1,
                        ) {
                            _serde::__private::Ok(__val) => __val,
                            _serde::__private::Err(__err) => {
                                return _serde::__private::Err(__err);
                            }
                        };
                        match _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "distribution",
                            distribution,
                        ) {
                            _serde::__private::Ok(__val) => __val,
                            _serde::__private::Err(__err) => {
                                return _serde::__private::Err(__err);
                            }
                        };
                        match _serde::ser::SerializeStructVariant::serialize_field(
                            &mut __serde_state,
                            "inv_coeffs",
                            inv_coeffs,
                        ) {
                            _serde::__private::Ok(__val) => __val,
                            _serde::__private::Err(__err) => {
                                return _serde::__private::Err(__err);
                            }
                        };
                        _serde::ser::SerializeStructVariant::end(__serde_state)
                    }
                }
            }
        }
    };
    impl GaussianDistribution {
        pub fn sample_n<R: Rng>(&mut self, n: usize, rng: &mut R) -> Vec<Array1<NNVFloat>> {
            match self {
                GaussianDistribution::TruncGaussian {
                    distribution,
                    inv_coeffs,
                } => {
                    let sample_arr = distribution.sample_n(n, rng);
                    sample_arr
                        .rows()
                        .into_iter()
                        .map(|x| (inv_coeffs.dot(&x.mapv(Into::into))))
                        .collect()
                }
                GaussianDistribution::Gaussian { ref loc, ref scale } => {
                    let samples = (Array2::random((n, loc.len()), StandardNormal).mapv(|x: f64| x)
                        * scale)
                        + loc;
                    samples.rows().into_iter().map(|x| x.to_owned()).collect()
                }
            }
        }
        pub fn cdf<R: Rng>(&mut self, n: usize, rng: &mut R) -> NNVFloat {
            match self {
                GaussianDistribution::TruncGaussian { distribution, .. } => {
                    let (est, _rel_err, _upper_bound) = distribution.cdf(n, rng);
                    est
                }
                GaussianDistribution::Gaussian { .. } => NNVFloat::one(),
            }
        }
        pub const fn try_get_tilting_solution(&self) -> Option<&TiltingSolution> {
            match self {
                GaussianDistribution::TruncGaussian { distribution, .. } => {
                    distribution.try_get_tilting_solution()
                }
                GaussianDistribution::Gaussian { .. } => None,
            }
        }
        pub fn populate_tilting_solution(&mut self, initialization: Option<&TiltingSolution>) {
            if let GaussianDistribution::TruncGaussian { distribution, .. } = self {
                distribution.get_tilting_solution(initialization);
            }
        }
    }
}
pub mod graph {
    #![allow(clippy::module_name_repetitions)]
    mod builder {}
    mod execute_engine {
        //! ## Engine lifecycle
        //! 1. Create the `Engine` by passing it a `Graph`
        //! 2. Call `run` or a variant to transform input representations to output representations
        //! 3. GOTO 2
        //!
        //! Calling `run` requires a closure or an `OperationVisitor` trait object. This visitor will use
        //! the operation data and input representation data to calculate output representations. It's
        //! setup like this to facilitate the use of new representations.
        use super::graph::{Graph, GraphState, OperationId, OperationNode, RepresentationId};
        use super::operation::Operation;
        use super::{GraphError, PhysicalOp};
        use std::collections::HashMap;
        use std::collections::HashSet;
        use std::fmt::Debug;
        pub struct Engine<'a> {
            graph: &'a Graph,
        }
        impl<'a> Engine<'a> {
            pub const fn new(graph: &'a Graph) -> Self {
                Self { graph }
            }
            /// Calculates output representations of a sub-graph, given the input representations and a visitor.
            ///
            /// # Description
            ///
            /// This function works with both visitors of a node and steps within a node.
            /// For instance, ReLU layers of a network are broken down into a step ReLU at
            /// each dimension. It is assumed that intermediate outputs of node steps take
            /// the same form as nodes themselves.
            ///
            /// # Arguments
            ///
            /// * `output_ids` - Ids of the representations to calculate.
            /// * `inputs` - Set of starting inputs required to calculate outputs
            ///     nodal visit call is replaced with the step visits.
            /// * `visit` - Performs the intermediate calculations at each node.
            ///     * Arguments:
            ///         * `op` - The operation of the visited node
            ///         * `inputs` - The input representations
            ///         * `step` - The step to calculate. `None` is passed in when inputs to the node are given.
            ///     * Returns:
            ///         * `new_step` - The step that was just calculated. Use `None` to signify that the operation is complete
            ///         * `repr` - The representation that was calculated.
            ///
            /// # Returns
            ///
            /// * `outputs` - The outputs for each id in `output_ids`
            ///
            /// # Errors
            /// TODO
            ///
            /// # Panics
            /// TODO
            pub fn run<T: Clone + Debug>(
                &self,
                output_ids: Vec<RepresentationId>,
                inputs: &[(RepresentationId, T)],
                mut visit: impl FnMut(&PhysicalOp, &Vec<&T>, Option<usize>) -> (Option<usize>, Vec<T>),
            ) -> Result<Vec<(RepresentationId, T)>, ExecuteError> {
                self.run_nodal(
                    output_ids,
                    inputs,
                    |_, op_node, inputs, step| -> (Option<usize>, Vec<T>) {
                        visit(op_node.get_operation(), inputs, step)
                    },
                )
            }
            pub fn run_nodal<T: Clone + Debug>(
                &self,
                output_ids: Vec<RepresentationId>,
                inputs: &[(RepresentationId, T)],
                mut visit: impl FnMut(
                    OperationId,
                    &OperationNode,
                    &Vec<&T>,
                    Option<usize>,
                ) -> (Option<usize>, Vec<T>),
            ) -> Result<Vec<(RepresentationId, T)>, ExecuteError> {
                let mut state = ExecutionState::<T>::default();
                let input_ids = inputs.iter().map(|&(id, _)| id).collect::<Vec<_>>();
                let operation_set = self.graph.get_operation_set(&output_ids, &input_ids)?;
                let mut op_node_vec: Vec<OperationId> = operation_set.into_iter().collect();
                op_node_vec.sort_unstable();
                if inputs
                    .iter()
                    .map(|(id, v)| state.set_representation(*id, v.clone()))
                    .any(|x| x.is_err())
                {
                    return Err(ExecuteError::GenericError);
                }
                for op_id in op_node_vec {
                    let op_node = self
                        .graph
                        .get_operation_node(&op_id)
                        .ok_or(ExecuteError::OperationNotExist { op_id })?;
                    let op = op_node.get_operation();
                    let mut input_ids = op_node
                        .get_output_ids()
                        .iter()
                        .map(|out_id| state.get_step_start(*out_id).copied())
                        .collect::<Option<Vec<_>>>()
                        .unwrap_or_else(|| op_node.get_input_ids().clone());
                    let mut step = input_ids.first().unwrap().operation_step;
                    loop {
                        let (new_step, outputs) = {
                            let reprs = input_ids
                                .iter()
                                .map(|&id| state.get_representation(id))
                                .collect::<Option<Vec<&T>>>()
                                .ok_or(ExecuteError::OneOfRepresentationsNotExist {
                                    repr_ids: op_node.get_input_ids().clone(),
                                })?;
                            visit(op_id, op_node, &reprs, step)
                        };
                        if outputs.len() != op_node.get_output_ids().len() {
                            return Err(ExecuteError::IncorrectOutputsFromVisitor {
                                expected: outputs.len(),
                                given: op_node.get_output_ids().len(),
                            });
                        }
                        for (&repr_id, repr) in
                            op_node.get_output_ids().iter().zip(outputs.into_iter())
                        {
                            let mut repr_id = repr_id;
                            repr_id.operation_step = new_step;
                            state.set_representation(repr_id, repr)?;
                        }
                        for &repr_id in op_node.get_output_ids() {
                            if let Some(&done_step) = state.get_step_end(repr_id) {
                                if new_step.map_or(true, |s| s > done_step) {
                                    return Err(ExecuteError::SkippedEndStepOfOperation {
                                        repr_id,
                                        done_step,
                                    });
                                }
                                if let Some (new_s) = new_step && new_s >= op . num_steps () . unwrap () - 1 { return Err (ExecuteError :: NewStepLargerThanNumSteps { new_step : new_s , last_step : op . num_steps () . unwrap () - 1 , }) ; }
                            }
                        }
                        if new_step.is_none() {
                            break;
                        }
                        step = new_step;
                        if step.unwrap() == 0 {
                            input_ids = op_node.get_output_ids().clone();
                        }
                        input_ids = input_ids.into_iter().map(|id| id.with_step(step)).collect();
                        for &repr_id in op_node.get_output_ids() {
                            if let Some(&max_step) = state.get_step_end(repr_id) {
                                if new_step.map_or(false, |s| s == max_step) {
                                    break;
                                }
                            }
                        }
                    }
                }
                let outputs = output_ids
                    .iter()
                    .map(|&id| state.get_representation(id).cloned().map(|r| (id, r)))
                    .collect::<Option<Vec<(RepresentationId, T)>>>()
                    .ok_or(ExecuteError::OneOfRepresentationsNotExist {
                        repr_ids: output_ids,
                    })?;
                Ok(outputs)
            }
            /// # Panics
            /// TODO
            pub fn run_graph_state_to<T: Clone + Debug>(
                &self,
                _state: &GraphState<T>,
                _repr_id: RepresentationId,
            ) -> GraphState<T> {
                ::core::panicking::panic("not yet implemented")
            }
            /// Calculates the output of a visitor given the inputs
            ///
            /// # Arguments
            ///
            /// * `outputs` - Set of representations to calculate
            /// * `inputs` - Set of starting inputs required to calculate outputs
            /// * `visitor` - Performs the intermediate calculations at each node
            ///
            /// # Panics
            /// TODO
            ///
            /// # Errors
            /// TODO
            pub fn run_node_visitor<T: Clone + Debug>(
                &self,
                _output_ids: &[RepresentationId],
                _inputs: &[(RepresentationId, T)],
                _visitor: &mut dyn OperationVisitor<T>,
            ) -> Result<Vec<(RepresentationId, T)>, ExecuteError> {
                ::core::panicking::panic("not yet implemented");
            }
        }
        pub enum ExecuteError {
            GenericError,
            PoppedEmptyStack,
            GraphError {
                err: GraphError,
            },
            IncorrectOutputsFromVisitor {
                expected: usize,
                given: usize,
            },
            NoOpCreatesRepresentation {
                repr_id: RepresentationId,
            },
            OperationAddedTwice {
                op_id: OperationId,
            },
            OperationNotExist {
                op_id: OperationId,
            },
            AnotherOpProducesOutput {
                op_id: OperationId,
            },
            OneOfRepresentationsNotExist {
                repr_ids: Vec<RepresentationId>,
            },
            ReprIdGivenByInputsAndOpState {
                repr_ids: HashSet<RepresentationId>,
            },
            VisitStepsTrueForSteppedInputsOrOutputs,
            SkippedEndStepOfOperation {
                repr_id: RepresentationId,
                done_step: usize,
            },
            NewStepLargerThanNumSteps {
                new_step: usize,
                last_step: usize,
            },
            StateAlreadyHasRepresentation {
                rep_id: RepresentationId,
            },
        }
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl ::core::fmt::Debug for ExecuteError {
            fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                match (&*self,) {
                    (&ExecuteError::GenericError,) => {
                        ::core::fmt::Formatter::write_str(f, "GenericError")
                    }
                    (&ExecuteError::PoppedEmptyStack,) => {
                        ::core::fmt::Formatter::write_str(f, "PoppedEmptyStack")
                    }
                    (&ExecuteError::GraphError { err: ref __self_0 },) => {
                        let debug_trait_builder =
                            &mut ::core::fmt::Formatter::debug_struct(f, "GraphError");
                        let _ = ::core::fmt::DebugStruct::field(
                            debug_trait_builder,
                            "err",
                            &&(*__self_0),
                        );
                        ::core::fmt::DebugStruct::finish(debug_trait_builder)
                    }
                    (&ExecuteError::IncorrectOutputsFromVisitor {
                        expected: ref __self_0,
                        given: ref __self_1,
                    },) => {
                        let debug_trait_builder = &mut ::core::fmt::Formatter::debug_struct(
                            f,
                            "IncorrectOutputsFromVisitor",
                        );
                        let _ = ::core::fmt::DebugStruct::field(
                            debug_trait_builder,
                            "expected",
                            &&(*__self_0),
                        );
                        let _ = ::core::fmt::DebugStruct::field(
                            debug_trait_builder,
                            "given",
                            &&(*__self_1),
                        );
                        ::core::fmt::DebugStruct::finish(debug_trait_builder)
                    }
                    (&ExecuteError::NoOpCreatesRepresentation {
                        repr_id: ref __self_0,
                    },) => {
                        let debug_trait_builder = &mut ::core::fmt::Formatter::debug_struct(
                            f,
                            "NoOpCreatesRepresentation",
                        );
                        let _ = ::core::fmt::DebugStruct::field(
                            debug_trait_builder,
                            "repr_id",
                            &&(*__self_0),
                        );
                        ::core::fmt::DebugStruct::finish(debug_trait_builder)
                    }
                    (&ExecuteError::OperationAddedTwice {
                        op_id: ref __self_0,
                    },) => {
                        let debug_trait_builder =
                            &mut ::core::fmt::Formatter::debug_struct(f, "OperationAddedTwice");
                        let _ = ::core::fmt::DebugStruct::field(
                            debug_trait_builder,
                            "op_id",
                            &&(*__self_0),
                        );
                        ::core::fmt::DebugStruct::finish(debug_trait_builder)
                    }
                    (&ExecuteError::OperationNotExist {
                        op_id: ref __self_0,
                    },) => {
                        let debug_trait_builder =
                            &mut ::core::fmt::Formatter::debug_struct(f, "OperationNotExist");
                        let _ = ::core::fmt::DebugStruct::field(
                            debug_trait_builder,
                            "op_id",
                            &&(*__self_0),
                        );
                        ::core::fmt::DebugStruct::finish(debug_trait_builder)
                    }
                    (&ExecuteError::AnotherOpProducesOutput {
                        op_id: ref __self_0,
                    },) => {
                        let debug_trait_builder =
                            &mut ::core::fmt::Formatter::debug_struct(f, "AnotherOpProducesOutput");
                        let _ = ::core::fmt::DebugStruct::field(
                            debug_trait_builder,
                            "op_id",
                            &&(*__self_0),
                        );
                        ::core::fmt::DebugStruct::finish(debug_trait_builder)
                    }
                    (&ExecuteError::OneOfRepresentationsNotExist {
                        repr_ids: ref __self_0,
                    },) => {
                        let debug_trait_builder = &mut ::core::fmt::Formatter::debug_struct(
                            f,
                            "OneOfRepresentationsNotExist",
                        );
                        let _ = ::core::fmt::DebugStruct::field(
                            debug_trait_builder,
                            "repr_ids",
                            &&(*__self_0),
                        );
                        ::core::fmt::DebugStruct::finish(debug_trait_builder)
                    }
                    (&ExecuteError::ReprIdGivenByInputsAndOpState {
                        repr_ids: ref __self_0,
                    },) => {
                        let debug_trait_builder = &mut ::core::fmt::Formatter::debug_struct(
                            f,
                            "ReprIdGivenByInputsAndOpState",
                        );
                        let _ = ::core::fmt::DebugStruct::field(
                            debug_trait_builder,
                            "repr_ids",
                            &&(*__self_0),
                        );
                        ::core::fmt::DebugStruct::finish(debug_trait_builder)
                    }
                    (&ExecuteError::VisitStepsTrueForSteppedInputsOrOutputs,) => {
                        ::core::fmt::Formatter::write_str(
                            f,
                            "VisitStepsTrueForSteppedInputsOrOutputs",
                        )
                    }
                    (&ExecuteError::SkippedEndStepOfOperation {
                        repr_id: ref __self_0,
                        done_step: ref __self_1,
                    },) => {
                        let debug_trait_builder = &mut ::core::fmt::Formatter::debug_struct(
                            f,
                            "SkippedEndStepOfOperation",
                        );
                        let _ = ::core::fmt::DebugStruct::field(
                            debug_trait_builder,
                            "repr_id",
                            &&(*__self_0),
                        );
                        let _ = ::core::fmt::DebugStruct::field(
                            debug_trait_builder,
                            "done_step",
                            &&(*__self_1),
                        );
                        ::core::fmt::DebugStruct::finish(debug_trait_builder)
                    }
                    (&ExecuteError::NewStepLargerThanNumSteps {
                        new_step: ref __self_0,
                        last_step: ref __self_1,
                    },) => {
                        let debug_trait_builder = &mut ::core::fmt::Formatter::debug_struct(
                            f,
                            "NewStepLargerThanNumSteps",
                        );
                        let _ = ::core::fmt::DebugStruct::field(
                            debug_trait_builder,
                            "new_step",
                            &&(*__self_0),
                        );
                        let _ = ::core::fmt::DebugStruct::field(
                            debug_trait_builder,
                            "last_step",
                            &&(*__self_1),
                        );
                        ::core::fmt::DebugStruct::finish(debug_trait_builder)
                    }
                    (&ExecuteError::StateAlreadyHasRepresentation {
                        rep_id: ref __self_0,
                    },) => {
                        let debug_trait_builder = &mut ::core::fmt::Formatter::debug_struct(
                            f,
                            "StateAlreadyHasRepresentation",
                        );
                        let _ = ::core::fmt::DebugStruct::field(
                            debug_trait_builder,
                            "rep_id",
                            &&(*__self_0),
                        );
                        ::core::fmt::DebugStruct::finish(debug_trait_builder)
                    }
                }
            }
        }
        impl From<GraphError> for ExecuteError {
            fn from(err: GraphError) -> Self {
                Self::GraphError { err }
            }
        }
        pub trait OperationVisitor<T: Clone> {
            /// Visits the operations in set topological order. Will always visit child operations before parent operations.
            ///
            /// # Arguments
            ///
            /// * `operation` - The operation being visited
            /// * `inputs` - Inputs to the operation
            fn visit(&mut self, operation: &PhysicalOp, inputs: Vec<&T>) -> Vec<T>;
        }
        pub struct ExecutionState<T: Clone> {
            /// Keeps track of all representations currently in memory
            representations: HashMap<RepresentationId, T>,
            /// Keeps track of which representations start at a step instead of directly from input
            step_starts: HashMap<usize, RepresentationId>,
            /// Keeps track of which representation ids signal the end of calculation within the node, i.e., shortcutting the computation.
            /// Maps `representation_node_id` -> `step`
            step_ends: HashMap<usize, usize>,
        }
        impl<T: Clone> ExecutionState<T> {
            pub fn new(
                representations: HashMap<RepresentationId, T>,
                step_starts: &[RepresentationId],
                step_ends: &[RepresentationId],
            ) -> Self {
                Self {
                    representations,
                    step_starts: step_starts
                        .iter()
                        .map(|&repr_id| (repr_id.representation_node_id, repr_id))
                        .collect(),
                    step_ends: step_ends
                        .iter()
                        .map(|repr_id| {
                            (
                                repr_id.representation_node_id,
                                repr_id.operation_step.unwrap(),
                            )
                        })
                        .collect(),
                }
            }
            pub fn get_step_start(&self, repr_id: RepresentationId) -> Option<&RepresentationId> {
                self.step_starts.get(&repr_id.representation_node_id)
            }
            pub fn get_step_end(&self, repr_id: RepresentationId) -> Option<&usize> {
                self.step_ends.get(&repr_id.representation_node_id)
            }
        }
        impl<T: Clone> Default for ExecutionState<T> {
            fn default() -> Self {
                Self {
                    representations: HashMap::new(),
                    step_starts: HashMap::new(),
                    step_ends: HashMap::new(),
                }
            }
        }
        impl<T: Clone> ExecutionState<T> {
            pub fn get_representation(&self, representation_id: RepresentationId) -> Option<&T> {
                self.representations.get(&representation_id)
            }
            pub fn set_representation(
                &mut self,
                representation_id: RepresentationId,
                representation: T,
            ) -> Result<(), ExecuteError> {
                if let std::collections::hash_map::Entry::Vacant(e) =
                    self.representations.entry(representation_id)
                {
                    e.insert(representation);
                    Ok(())
                } else {
                    Err(ExecuteError::StateAlreadyHasRepresentation {
                        rep_id: representation_id,
                    })
                }
            }
        }
    }
    mod graph {
        use crate::dnn::conv::Conv;
        use crate::dnn::dense::Dense;
        use crate::dnn::interpolate::Interpolate;
        use crate::dnn::relu::ReLU;
        use enum_dispatch::enum_dispatch;
        use itertools::Itertools;
        use serde::{Deserialize, Serialize};
        use std::collections::{HashMap, HashSet};
        use std::fmt::{self, Debug};
        /// Unique key for an operation scoped to a Graph.
        pub type OperationId = usize;
        /// # Description
        ///
        /// Unique key for a representation scoped to a Graph. I.e., something that is input/output of the graph ops.
        /// E.g., tensors, stars, bounds.
        pub struct RepresentationId {
            pub representation_node_id: usize,
            /// If this representation is internal to a stepped operation (i.e. is produced and consumed by that operation),
            /// This field encodes the index of the step run to create this representation if this representation is intermediate,
            /// and should be None on the final operation output
            pub operation_step: Option<usize>,
        }
        impl ::core::marker::StructuralPartialEq for RepresentationId {}
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl ::core::cmp::PartialEq for RepresentationId {
            #[inline]
            fn eq(&self, other: &RepresentationId) -> bool {
                match *other {
                    RepresentationId {
                        representation_node_id: ref __self_1_0,
                        operation_step: ref __self_1_1,
                    } => match *self {
                        RepresentationId {
                            representation_node_id: ref __self_0_0,
                            operation_step: ref __self_0_1,
                        } => (*__self_0_0) == (*__self_1_0) && (*__self_0_1) == (*__self_1_1),
                    },
                }
            }
            #[inline]
            fn ne(&self, other: &RepresentationId) -> bool {
                match *other {
                    RepresentationId {
                        representation_node_id: ref __self_1_0,
                        operation_step: ref __self_1_1,
                    } => match *self {
                        RepresentationId {
                            representation_node_id: ref __self_0_0,
                            operation_step: ref __self_0_1,
                        } => (*__self_0_0) != (*__self_1_0) || (*__self_0_1) != (*__self_1_1),
                    },
                }
            }
        }
        impl ::core::marker::StructuralEq for RepresentationId {}
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl ::core::cmp::Eq for RepresentationId {
            #[inline]
            #[doc(hidden)]
            #[no_coverage]
            fn assert_receiver_is_total_eq(&self) -> () {
                {
                    let _: ::core::cmp::AssertParamIsEq<usize>;
                    let _: ::core::cmp::AssertParamIsEq<Option<usize>>;
                }
            }
        }
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl ::core::cmp::PartialOrd for RepresentationId {
            #[inline]
            fn partial_cmp(
                &self,
                other: &RepresentationId,
            ) -> ::core::option::Option<::core::cmp::Ordering> {
                match *other {
                    RepresentationId {
                        representation_node_id: ref __self_1_0,
                        operation_step: ref __self_1_1,
                    } => match *self {
                        RepresentationId {
                            representation_node_id: ref __self_0_0,
                            operation_step: ref __self_0_1,
                        } => match ::core::cmp::PartialOrd::partial_cmp(
                            &(*__self_0_0),
                            &(*__self_1_0),
                        ) {
                            ::core::option::Option::Some(::core::cmp::Ordering::Equal) => {
                                match ::core::cmp::PartialOrd::partial_cmp(
                                    &(*__self_0_1),
                                    &(*__self_1_1),
                                ) {
                                    ::core::option::Option::Some(::core::cmp::Ordering::Equal) => {
                                        ::core::option::Option::Some(::core::cmp::Ordering::Equal)
                                    }
                                    cmp => cmp,
                                }
                            }
                            cmp => cmp,
                        },
                    },
                }
            }
        }
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl ::core::cmp::Ord for RepresentationId {
            #[inline]
            fn cmp(&self, other: &RepresentationId) -> ::core::cmp::Ordering {
                match *other {
                    RepresentationId {
                        representation_node_id: ref __self_1_0,
                        operation_step: ref __self_1_1,
                    } => match *self {
                        RepresentationId {
                            representation_node_id: ref __self_0_0,
                            operation_step: ref __self_0_1,
                        } => match ::core::cmp::Ord::cmp(&(*__self_0_0), &(*__self_1_0)) {
                            ::core::cmp::Ordering::Equal => {
                                match ::core::cmp::Ord::cmp(&(*__self_0_1), &(*__self_1_1)) {
                                    ::core::cmp::Ordering::Equal => ::core::cmp::Ordering::Equal,
                                    cmp => cmp,
                                }
                            }
                            cmp => cmp,
                        },
                    },
                }
            }
        }
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl ::core::fmt::Debug for RepresentationId {
            fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                match *self {
                    RepresentationId {
                        representation_node_id: ref __self_0_0,
                        operation_step: ref __self_0_1,
                    } => {
                        let debug_trait_builder =
                            &mut ::core::fmt::Formatter::debug_struct(f, "RepresentationId");
                        let _ = ::core::fmt::DebugStruct::field(
                            debug_trait_builder,
                            "representation_node_id",
                            &&(*__self_0_0),
                        );
                        let _ = ::core::fmt::DebugStruct::field(
                            debug_trait_builder,
                            "operation_step",
                            &&(*__self_0_1),
                        );
                        ::core::fmt::DebugStruct::finish(debug_trait_builder)
                    }
                }
            }
        }
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl ::core::clone::Clone for RepresentationId {
            #[inline]
            fn clone(&self) -> RepresentationId {
                {
                    let _: ::core::clone::AssertParamIsClone<usize>;
                    let _: ::core::clone::AssertParamIsClone<Option<usize>>;
                    *self
                }
            }
        }
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl ::core::marker::Copy for RepresentationId {}
        #[doc(hidden)]
        #[allow(non_upper_case_globals, unused_attributes, unused_qualifications)]
        const _: () = {
            #[allow(unused_extern_crates, clippy::useless_attribute)]
            extern crate serde as _serde;
            #[automatically_derived]
            impl _serde::Serialize for RepresentationId {
                fn serialize<__S>(
                    &self,
                    __serializer: __S,
                ) -> _serde::__private::Result<__S::Ok, __S::Error>
                where
                    __S: _serde::Serializer,
                {
                    let mut __serde_state = match _serde::Serializer::serialize_struct(
                        __serializer,
                        "RepresentationId",
                        false as usize + 1 + 1,
                    ) {
                        _serde::__private::Ok(__val) => __val,
                        _serde::__private::Err(__err) => {
                            return _serde::__private::Err(__err);
                        }
                    };
                    match _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "representation_node_id",
                        &self.representation_node_id,
                    ) {
                        _serde::__private::Ok(__val) => __val,
                        _serde::__private::Err(__err) => {
                            return _serde::__private::Err(__err);
                        }
                    };
                    match _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "operation_step",
                        &self.operation_step,
                    ) {
                        _serde::__private::Ok(__val) => __val,
                        _serde::__private::Err(__err) => {
                            return _serde::__private::Err(__err);
                        }
                    };
                    _serde::ser::SerializeStruct::end(__serde_state)
                }
            }
        };
        #[doc(hidden)]
        #[allow(non_upper_case_globals, unused_attributes, unused_qualifications)]
        const _: () = {
            #[allow(unused_extern_crates, clippy::useless_attribute)]
            extern crate serde as _serde;
            #[automatically_derived]
            impl<'de> _serde::Deserialize<'de> for RepresentationId {
                fn deserialize<__D>(
                    __deserializer: __D,
                ) -> _serde::__private::Result<Self, __D::Error>
                where
                    __D: _serde::Deserializer<'de>,
                {
                    #[allow(non_camel_case_types)]
                    enum __Field {
                        __field0,
                        __field1,
                        __ignore,
                    }
                    struct __FieldVisitor;
                    impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                        type Value = __Field;
                        fn expecting(
                            &self,
                            __formatter: &mut _serde::__private::Formatter,
                        ) -> _serde::__private::fmt::Result {
                            _serde::__private::Formatter::write_str(__formatter, "field identifier")
                        }
                        fn visit_u64<__E>(
                            self,
                            __value: u64,
                        ) -> _serde::__private::Result<Self::Value, __E>
                        where
                            __E: _serde::de::Error,
                        {
                            match __value {
                                0u64 => _serde::__private::Ok(__Field::__field0),
                                1u64 => _serde::__private::Ok(__Field::__field1),
                                _ => _serde::__private::Ok(__Field::__ignore),
                            }
                        }
                        fn visit_str<__E>(
                            self,
                            __value: &str,
                        ) -> _serde::__private::Result<Self::Value, __E>
                        where
                            __E: _serde::de::Error,
                        {
                            match __value {
                                "representation_node_id" => {
                                    _serde::__private::Ok(__Field::__field0)
                                }
                                "operation_step" => _serde::__private::Ok(__Field::__field1),
                                _ => _serde::__private::Ok(__Field::__ignore),
                            }
                        }
                        fn visit_bytes<__E>(
                            self,
                            __value: &[u8],
                        ) -> _serde::__private::Result<Self::Value, __E>
                        where
                            __E: _serde::de::Error,
                        {
                            match __value {
                                b"representation_node_id" => {
                                    _serde::__private::Ok(__Field::__field0)
                                }
                                b"operation_step" => _serde::__private::Ok(__Field::__field1),
                                _ => _serde::__private::Ok(__Field::__ignore),
                            }
                        }
                    }
                    impl<'de> _serde::Deserialize<'de> for __Field {
                        #[inline]
                        fn deserialize<__D>(
                            __deserializer: __D,
                        ) -> _serde::__private::Result<Self, __D::Error>
                        where
                            __D: _serde::Deserializer<'de>,
                        {
                            _serde::Deserializer::deserialize_identifier(
                                __deserializer,
                                __FieldVisitor,
                            )
                        }
                    }
                    struct __Visitor<'de> {
                        marker: _serde::__private::PhantomData<RepresentationId>,
                        lifetime: _serde::__private::PhantomData<&'de ()>,
                    }
                    impl<'de> _serde::de::Visitor<'de> for __Visitor<'de> {
                        type Value = RepresentationId;
                        fn expecting(
                            &self,
                            __formatter: &mut _serde::__private::Formatter,
                        ) -> _serde::__private::fmt::Result {
                            _serde::__private::Formatter::write_str(
                                __formatter,
                                "struct RepresentationId",
                            )
                        }
                        #[inline]
                        fn visit_seq<__A>(
                            self,
                            mut __seq: __A,
                        ) -> _serde::__private::Result<Self::Value, __A::Error>
                        where
                            __A: _serde::de::SeqAccess<'de>,
                        {
                            let __field0 = match match _serde::de::SeqAccess::next_element::<usize>(
                                &mut __seq,
                            ) {
                                _serde::__private::Ok(__val) => __val,
                                _serde::__private::Err(__err) => {
                                    return _serde::__private::Err(__err);
                                }
                            } {
                                _serde::__private::Some(__value) => __value,
                                _serde::__private::None => {
                                    return _serde::__private::Err(
                                        _serde::de::Error::invalid_length(
                                            0usize,
                                            &"struct RepresentationId with 2 elements",
                                        ),
                                    );
                                }
                            };
                            let __field1 = match match _serde::de::SeqAccess::next_element::<
                                Option<usize>,
                            >(&mut __seq)
                            {
                                _serde::__private::Ok(__val) => __val,
                                _serde::__private::Err(__err) => {
                                    return _serde::__private::Err(__err);
                                }
                            } {
                                _serde::__private::Some(__value) => __value,
                                _serde::__private::None => {
                                    return _serde::__private::Err(
                                        _serde::de::Error::invalid_length(
                                            1usize,
                                            &"struct RepresentationId with 2 elements",
                                        ),
                                    );
                                }
                            };
                            _serde::__private::Ok(RepresentationId {
                                representation_node_id: __field0,
                                operation_step: __field1,
                            })
                        }
                        #[inline]
                        fn visit_map<__A>(
                            self,
                            mut __map: __A,
                        ) -> _serde::__private::Result<Self::Value, __A::Error>
                        where
                            __A: _serde::de::MapAccess<'de>,
                        {
                            let mut __field0: _serde::__private::Option<usize> =
                                _serde::__private::None;
                            let mut __field1: _serde::__private::Option<Option<usize>> =
                                _serde::__private::None;
                            while let _serde::__private::Some(__key) =
                                match _serde::de::MapAccess::next_key::<__Field>(&mut __map) {
                                    _serde::__private::Ok(__val) => __val,
                                    _serde::__private::Err(__err) => {
                                        return _serde::__private::Err(__err);
                                    }
                                }
                            {
                                match __key {
                                    __Field::__field0 => {
                                        if _serde::__private::Option::is_some(&__field0) {
                                            return _serde::__private::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field(
                                                    "representation_node_id",
                                                ),
                                            );
                                        }
                                        __field0 = _serde::__private::Some(
                                            match _serde::de::MapAccess::next_value::<usize>(
                                                &mut __map,
                                            ) {
                                                _serde::__private::Ok(__val) => __val,
                                                _serde::__private::Err(__err) => {
                                                    return _serde::__private::Err(__err);
                                                }
                                            },
                                        );
                                    }
                                    __Field::__field1 => {
                                        if _serde::__private::Option::is_some(&__field1) {
                                            return _serde::__private::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field(
                                                    "operation_step",
                                                ),
                                            );
                                        }
                                        __field1 = _serde::__private::Some(
                                            match _serde::de::MapAccess::next_value::<Option<usize>>(
                                                &mut __map,
                                            ) {
                                                _serde::__private::Ok(__val) => __val,
                                                _serde::__private::Err(__err) => {
                                                    return _serde::__private::Err(__err);
                                                }
                                            },
                                        );
                                    }
                                    _ => {
                                        let _ = match _serde::de::MapAccess::next_value::<
                                            _serde::de::IgnoredAny,
                                        >(
                                            &mut __map
                                        ) {
                                            _serde::__private::Ok(__val) => __val,
                                            _serde::__private::Err(__err) => {
                                                return _serde::__private::Err(__err);
                                            }
                                        };
                                    }
                                }
                            }
                            let __field0 = match __field0 {
                                _serde::__private::Some(__field0) => __field0,
                                _serde::__private::None => {
                                    match _serde::__private::de::missing_field(
                                        "representation_node_id",
                                    ) {
                                        _serde::__private::Ok(__val) => __val,
                                        _serde::__private::Err(__err) => {
                                            return _serde::__private::Err(__err);
                                        }
                                    }
                                }
                            };
                            let __field1 = match __field1 {
                                _serde::__private::Some(__field1) => __field1,
                                _serde::__private::None => {
                                    match _serde::__private::de::missing_field("operation_step") {
                                        _serde::__private::Ok(__val) => __val,
                                        _serde::__private::Err(__err) => {
                                            return _serde::__private::Err(__err);
                                        }
                                    }
                                }
                            };
                            _serde::__private::Ok(RepresentationId {
                                representation_node_id: __field0,
                                operation_step: __field1,
                            })
                        }
                    }
                    const FIELDS: &'static [&'static str] =
                        &["representation_node_id", "operation_step"];
                    _serde::Deserializer::deserialize_struct(
                        __deserializer,
                        "RepresentationId",
                        FIELDS,
                        __Visitor {
                            marker: _serde::__private::PhantomData::<RepresentationId>,
                            lifetime: _serde::__private::PhantomData,
                        },
                    )
                }
            }
        };
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl ::core::hash::Hash for RepresentationId {
            fn hash<__H: ::core::hash::Hasher>(&self, state: &mut __H) -> () {
                match *self {
                    RepresentationId {
                        representation_node_id: ref __self_0_0,
                        operation_step: ref __self_0_1,
                    } => {
                        ::core::hash::Hash::hash(&(*__self_0_0), state);
                        ::core::hash::Hash::hash(&(*__self_0_1), state)
                    }
                }
            }
        }
        impl RepresentationId {
            pub const fn new(representation_node_id: usize, operation_step: Option<usize>) -> Self {
                Self {
                    representation_node_id,
                    operation_step,
                }
            }
            #[must_use]
            pub const fn with_step(mut self, operation_step: Option<usize>) -> Self {
                self.operation_step = operation_step;
                self
            }
        }
        /// # Invariants:
        /// - Graph state has all the alive representations required to compute the output representations, including the output representations
        pub struct GraphState<T: Debug + Clone> {
            /// The representations being calculated
            pub output_representation_ids: Vec<RepresentationId>,
            /// All representations required to run the remaining operations, including final output representations
            pub alive_representations: HashMap<RepresentationId, T>,
            /// The number of operations that require each alive representations (`output_representation_ids` count)
            pub reference_counts: HashMap<RepresentationId, usize>,
        }
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl<T: ::core::fmt::Debug + Debug + Clone> ::core::fmt::Debug for GraphState<T> {
            fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                match *self {
                    GraphState {
                        output_representation_ids: ref __self_0_0,
                        alive_representations: ref __self_0_1,
                        reference_counts: ref __self_0_2,
                    } => {
                        let debug_trait_builder =
                            &mut ::core::fmt::Formatter::debug_struct(f, "GraphState");
                        let _ = ::core::fmt::DebugStruct::field(
                            debug_trait_builder,
                            "output_representation_ids",
                            &&(*__self_0_0),
                        );
                        let _ = ::core::fmt::DebugStruct::field(
                            debug_trait_builder,
                            "alive_representations",
                            &&(*__self_0_1),
                        );
                        let _ = ::core::fmt::DebugStruct::field(
                            debug_trait_builder,
                            "reference_counts",
                            &&(*__self_0_2),
                        );
                        ::core::fmt::DebugStruct::finish(debug_trait_builder)
                    }
                }
            }
        }
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl<T: ::core::clone::Clone + Debug + Clone> ::core::clone::Clone for GraphState<T> {
            #[inline]
            fn clone(&self) -> GraphState<T> {
                match *self {
                    GraphState {
                        output_representation_ids: ref __self_0_0,
                        alive_representations: ref __self_0_1,
                        reference_counts: ref __self_0_2,
                    } => GraphState {
                        output_representation_ids: ::core::clone::Clone::clone(&(*__self_0_0)),
                        alive_representations: ::core::clone::Clone::clone(&(*__self_0_1)),
                        reference_counts: ::core::clone::Clone::clone(&(*__self_0_2)),
                    },
                }
            }
        }
        #[doc(hidden)]
        #[allow(non_upper_case_globals, unused_attributes, unused_qualifications)]
        const _: () = {
            #[allow(unused_extern_crates, clippy::useless_attribute)]
            extern crate serde as _serde;
            #[automatically_derived]
            impl<'de, T: Debug + Clone> _serde::Deserialize<'de> for GraphState<T>
            where
                T: _serde::Deserialize<'de>,
            {
                fn deserialize<__D>(
                    __deserializer: __D,
                ) -> _serde::__private::Result<Self, __D::Error>
                where
                    __D: _serde::Deserializer<'de>,
                {
                    #[allow(non_camel_case_types)]
                    enum __Field {
                        __field0,
                        __field1,
                        __field2,
                        __ignore,
                    }
                    struct __FieldVisitor;
                    impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                        type Value = __Field;
                        fn expecting(
                            &self,
                            __formatter: &mut _serde::__private::Formatter,
                        ) -> _serde::__private::fmt::Result {
                            _serde::__private::Formatter::write_str(__formatter, "field identifier")
                        }
                        fn visit_u64<__E>(
                            self,
                            __value: u64,
                        ) -> _serde::__private::Result<Self::Value, __E>
                        where
                            __E: _serde::de::Error,
                        {
                            match __value {
                                0u64 => _serde::__private::Ok(__Field::__field0),
                                1u64 => _serde::__private::Ok(__Field::__field1),
                                2u64 => _serde::__private::Ok(__Field::__field2),
                                _ => _serde::__private::Ok(__Field::__ignore),
                            }
                        }
                        fn visit_str<__E>(
                            self,
                            __value: &str,
                        ) -> _serde::__private::Result<Self::Value, __E>
                        where
                            __E: _serde::de::Error,
                        {
                            match __value {
                                "output_representation_ids" => {
                                    _serde::__private::Ok(__Field::__field0)
                                }
                                "alive_representations" => _serde::__private::Ok(__Field::__field1),
                                "reference_counts" => _serde::__private::Ok(__Field::__field2),
                                _ => _serde::__private::Ok(__Field::__ignore),
                            }
                        }
                        fn visit_bytes<__E>(
                            self,
                            __value: &[u8],
                        ) -> _serde::__private::Result<Self::Value, __E>
                        where
                            __E: _serde::de::Error,
                        {
                            match __value {
                                b"output_representation_ids" => {
                                    _serde::__private::Ok(__Field::__field0)
                                }
                                b"alive_representations" => {
                                    _serde::__private::Ok(__Field::__field1)
                                }
                                b"reference_counts" => _serde::__private::Ok(__Field::__field2),
                                _ => _serde::__private::Ok(__Field::__ignore),
                            }
                        }
                    }
                    impl<'de> _serde::Deserialize<'de> for __Field {
                        #[inline]
                        fn deserialize<__D>(
                            __deserializer: __D,
                        ) -> _serde::__private::Result<Self, __D::Error>
                        where
                            __D: _serde::Deserializer<'de>,
                        {
                            _serde::Deserializer::deserialize_identifier(
                                __deserializer,
                                __FieldVisitor,
                            )
                        }
                    }
                    struct __Visitor<'de, T: Debug + Clone>
                    where
                        T: _serde::Deserialize<'de>,
                    {
                        marker: _serde::__private::PhantomData<GraphState<T>>,
                        lifetime: _serde::__private::PhantomData<&'de ()>,
                    }
                    impl<'de, T: Debug + Clone> _serde::de::Visitor<'de> for __Visitor<'de, T>
                    where
                        T: _serde::Deserialize<'de>,
                    {
                        type Value = GraphState<T>;
                        fn expecting(
                            &self,
                            __formatter: &mut _serde::__private::Formatter,
                        ) -> _serde::__private::fmt::Result {
                            _serde::__private::Formatter::write_str(
                                __formatter,
                                "struct GraphState",
                            )
                        }
                        #[inline]
                        fn visit_seq<__A>(
                            self,
                            mut __seq: __A,
                        ) -> _serde::__private::Result<Self::Value, __A::Error>
                        where
                            __A: _serde::de::SeqAccess<'de>,
                        {
                            let __field0 = match match _serde::de::SeqAccess::next_element::<
                                Vec<RepresentationId>,
                            >(&mut __seq)
                            {
                                _serde::__private::Ok(__val) => __val,
                                _serde::__private::Err(__err) => {
                                    return _serde::__private::Err(__err);
                                }
                            } {
                                _serde::__private::Some(__value) => __value,
                                _serde::__private::None => {
                                    return _serde::__private::Err(
                                        _serde::de::Error::invalid_length(
                                            0usize,
                                            &"struct GraphState with 3 elements",
                                        ),
                                    );
                                }
                            };
                            let __field1 = match match _serde::de::SeqAccess::next_element::<
                                HashMap<RepresentationId, T>,
                            >(&mut __seq)
                            {
                                _serde::__private::Ok(__val) => __val,
                                _serde::__private::Err(__err) => {
                                    return _serde::__private::Err(__err);
                                }
                            } {
                                _serde::__private::Some(__value) => __value,
                                _serde::__private::None => {
                                    return _serde::__private::Err(
                                        _serde::de::Error::invalid_length(
                                            1usize,
                                            &"struct GraphState with 3 elements",
                                        ),
                                    );
                                }
                            };
                            let __field2 = match match _serde::de::SeqAccess::next_element::<
                                HashMap<RepresentationId, usize>,
                            >(&mut __seq)
                            {
                                _serde::__private::Ok(__val) => __val,
                                _serde::__private::Err(__err) => {
                                    return _serde::__private::Err(__err);
                                }
                            } {
                                _serde::__private::Some(__value) => __value,
                                _serde::__private::None => {
                                    return _serde::__private::Err(
                                        _serde::de::Error::invalid_length(
                                            2usize,
                                            &"struct GraphState with 3 elements",
                                        ),
                                    );
                                }
                            };
                            _serde::__private::Ok(GraphState {
                                output_representation_ids: __field0,
                                alive_representations: __field1,
                                reference_counts: __field2,
                            })
                        }
                        #[inline]
                        fn visit_map<__A>(
                            self,
                            mut __map: __A,
                        ) -> _serde::__private::Result<Self::Value, __A::Error>
                        where
                            __A: _serde::de::MapAccess<'de>,
                        {
                            let mut __field0: _serde::__private::Option<Vec<RepresentationId>> =
                                _serde::__private::None;
                            let mut __field1: _serde::__private::Option<
                                HashMap<RepresentationId, T>,
                            > = _serde::__private::None;
                            let mut __field2: _serde::__private::Option<
                                HashMap<RepresentationId, usize>,
                            > = _serde::__private::None;
                            while let _serde::__private::Some(__key) =
                                match _serde::de::MapAccess::next_key::<__Field>(&mut __map) {
                                    _serde::__private::Ok(__val) => __val,
                                    _serde::__private::Err(__err) => {
                                        return _serde::__private::Err(__err);
                                    }
                                }
                            {
                                match __key {
                                    __Field::__field0 => {
                                        if _serde::__private::Option::is_some(&__field0) {
                                            return _serde::__private::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field(
                                                    "output_representation_ids",
                                                ),
                                            );
                                        }
                                        __field0 = _serde::__private::Some(
                                            match _serde::de::MapAccess::next_value::<
                                                Vec<RepresentationId>,
                                            >(
                                                &mut __map
                                            ) {
                                                _serde::__private::Ok(__val) => __val,
                                                _serde::__private::Err(__err) => {
                                                    return _serde::__private::Err(__err);
                                                }
                                            },
                                        );
                                    }
                                    __Field::__field1 => {
                                        if _serde::__private::Option::is_some(&__field1) {
                                            return _serde::__private::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field(
                                                    "alive_representations",
                                                ),
                                            );
                                        }
                                        __field1 = _serde::__private::Some(
                                            match _serde::de::MapAccess::next_value::<
                                                HashMap<RepresentationId, T>,
                                            >(
                                                &mut __map
                                            ) {
                                                _serde::__private::Ok(__val) => __val,
                                                _serde::__private::Err(__err) => {
                                                    return _serde::__private::Err(__err);
                                                }
                                            },
                                        );
                                    }
                                    __Field::__field2 => {
                                        if _serde::__private::Option::is_some(&__field2) {
                                            return _serde::__private::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field(
                                                    "reference_counts",
                                                ),
                                            );
                                        }
                                        __field2 = _serde::__private::Some(
                                            match _serde::de::MapAccess::next_value::<
                                                HashMap<RepresentationId, usize>,
                                            >(
                                                &mut __map
                                            ) {
                                                _serde::__private::Ok(__val) => __val,
                                                _serde::__private::Err(__err) => {
                                                    return _serde::__private::Err(__err);
                                                }
                                            },
                                        );
                                    }
                                    _ => {
                                        let _ = match _serde::de::MapAccess::next_value::<
                                            _serde::de::IgnoredAny,
                                        >(
                                            &mut __map
                                        ) {
                                            _serde::__private::Ok(__val) => __val,
                                            _serde::__private::Err(__err) => {
                                                return _serde::__private::Err(__err);
                                            }
                                        };
                                    }
                                }
                            }
                            let __field0 = match __field0 {
                                _serde::__private::Some(__field0) => __field0,
                                _serde::__private::None => {
                                    match _serde::__private::de::missing_field(
                                        "output_representation_ids",
                                    ) {
                                        _serde::__private::Ok(__val) => __val,
                                        _serde::__private::Err(__err) => {
                                            return _serde::__private::Err(__err);
                                        }
                                    }
                                }
                            };
                            let __field1 = match __field1 {
                                _serde::__private::Some(__field1) => __field1,
                                _serde::__private::None => {
                                    match _serde::__private::de::missing_field(
                                        "alive_representations",
                                    ) {
                                        _serde::__private::Ok(__val) => __val,
                                        _serde::__private::Err(__err) => {
                                            return _serde::__private::Err(__err);
                                        }
                                    }
                                }
                            };
                            let __field2 = match __field2 {
                                _serde::__private::Some(__field2) => __field2,
                                _serde::__private::None => {
                                    match _serde::__private::de::missing_field("reference_counts") {
                                        _serde::__private::Ok(__val) => __val,
                                        _serde::__private::Err(__err) => {
                                            return _serde::__private::Err(__err);
                                        }
                                    }
                                }
                            };
                            _serde::__private::Ok(GraphState {
                                output_representation_ids: __field0,
                                alive_representations: __field1,
                                reference_counts: __field2,
                            })
                        }
                    }
                    const FIELDS: &'static [&'static str] = &[
                        "output_representation_ids",
                        "alive_representations",
                        "reference_counts",
                    ];
                    _serde::Deserializer::deserialize_struct(
                        __deserializer,
                        "GraphState",
                        FIELDS,
                        __Visitor {
                            marker: _serde::__private::PhantomData::<GraphState<T>>,
                            lifetime: _serde::__private::PhantomData,
                        },
                    )
                }
            }
        };
        #[doc(hidden)]
        #[allow(non_upper_case_globals, unused_attributes, unused_qualifications)]
        const _: () = {
            #[allow(unused_extern_crates, clippy::useless_attribute)]
            extern crate serde as _serde;
            #[automatically_derived]
            impl<T: Debug + Clone> _serde::Serialize for GraphState<T>
            where
                T: _serde::Serialize,
            {
                fn serialize<__S>(
                    &self,
                    __serializer: __S,
                ) -> _serde::__private::Result<__S::Ok, __S::Error>
                where
                    __S: _serde::Serializer,
                {
                    let mut __serde_state = match _serde::Serializer::serialize_struct(
                        __serializer,
                        "GraphState",
                        false as usize + 1 + 1 + 1,
                    ) {
                        _serde::__private::Ok(__val) => __val,
                        _serde::__private::Err(__err) => {
                            return _serde::__private::Err(__err);
                        }
                    };
                    match _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "output_representation_ids",
                        &self.output_representation_ids,
                    ) {
                        _serde::__private::Ok(__val) => __val,
                        _serde::__private::Err(__err) => {
                            return _serde::__private::Err(__err);
                        }
                    };
                    match _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "alive_representations",
                        &self.alive_representations,
                    ) {
                        _serde::__private::Ok(__val) => __val,
                        _serde::__private::Err(__err) => {
                            return _serde::__private::Err(__err);
                        }
                    };
                    match _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "reference_counts",
                        &self.reference_counts,
                    ) {
                        _serde::__private::Ok(__val) => __val,
                        _serde::__private::Err(__err) => {
                            return _serde::__private::Err(__err);
                        }
                    };
                    _serde::ser::SerializeStruct::end(__serde_state)
                }
            }
        };
        impl<T: Debug + Clone> GraphState<T> {
            pub fn new(
                output_ids: Vec<RepresentationId>,
                input_representations: HashMap<RepresentationId, T>,
                graph: &Graph,
            ) -> Self {
                let input_representation_ids = input_representations
                    .iter()
                    .map(|(&repr_id, _)| repr_id)
                    .collect::<Vec<_>>();
                let reference_counts =
                    graph.get_reference_counts(&output_ids, &input_representation_ids);
                Self {
                    output_representation_ids: output_ids,
                    alive_representations: input_representations,
                    reference_counts,
                }
            }
        }
        /// A topo-sorted list of operations that transforms representations into representations.
        pub struct Graph {
            representation_ops: HashMap<RepresentationId, OperationId>,
            operation_nodes: Vec<OperationNode>,
        }
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl ::core::default::Default for Graph {
            #[inline]
            fn default() -> Graph {
                Graph {
                    representation_ops: ::core::default::Default::default(),
                    operation_nodes: ::core::default::Default::default(),
                }
            }
        }
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl ::core::clone::Clone for Graph {
            #[inline]
            fn clone(&self) -> Graph {
                match *self {
                    Graph {
                        representation_ops: ref __self_0_0,
                        operation_nodes: ref __self_0_1,
                    } => Graph {
                        representation_ops: ::core::clone::Clone::clone(&(*__self_0_0)),
                        operation_nodes: ::core::clone::Clone::clone(&(*__self_0_1)),
                    },
                }
            }
        }
        impl Graph {
            /// Get the specific id of the operation that produces a specific representation
            ///
            /// # Arguments
            ///
            /// * `id` - Representation whose producing operation we are trying to retrieve
            pub fn get_representation_op_id(&self, id: &RepresentationId) -> Option<OperationId> {
                self.representation_ops.get(id).copied()
            }
            /// Get the ids of the operations that `id` feeds into.
            ///
            /// # Description
            ///
            /// If an `id` has no `operation_step`, then the operation that has `id` as an input representation id is returned.
            /// If `id` has an `operation_step`, then *the* operation that produces an identical id with no operation step is returned.
            pub fn get_representation_input_op_ids(
                &self,
                id: &RepresentationId,
            ) -> Vec<OperationId> {
                if id.operation_step.is_none() {
                    self.operation_nodes
                        .iter()
                        .enumerate()
                        .filter(|(_, op_node)| op_node.get_input_ids().contains(id))
                        .map(|(idx, _)| idx)
                        .collect::<Vec<_>>()
                } else {
                    let mut id_no_step = *id;
                    id_no_step.operation_step = None;
                    self.operation_nodes
                        .iter()
                        .enumerate()
                        .filter(|(_, op_node)| op_node.get_output_ids().contains(&id_no_step))
                        .map(|(idx, _)| idx)
                        .collect::<Vec<_>>()
                }
            }
            pub fn get_operations(&self) -> &Vec<OperationNode> {
                &self.operation_nodes
            }
            pub fn get_operation_node(&self, id: &OperationId) -> Option<&OperationNode> {
                self.operation_nodes.get(*id)
            }
            /// # Errors
            /// TODO
            pub fn add_operation(
                &mut self,
                op: PhysicalOp,
                inputs: Vec<RepresentationId>,
                outputs: Vec<RepresentationId>,
            ) -> Result<OperationId, GraphError> {
                let node = OperationNode::new(op, inputs, outputs);
                self.add_operation_node(node)
            }
            /// Add an operation node to the graph. Nodes should be added in a topological order
            ///
            /// # Arguments
            ///
            /// * `node`: The node to add
            ///
            /// # Errors
            /// TODO
            pub fn add_operation_node(
                &mut self,
                node: OperationNode,
            ) -> Result<OperationId, GraphError> {
                let node_id = self.operation_nodes.len();
                if let Some(op_id) = node
                    .get_output_ids()
                    .iter()
                    .map(|&id| self.representation_ops.insert(id, node_id))
                    .filter(|&old_id| old_id.is_some())
                    .map(std::option::Option::unwrap)
                    .next()
                {
                    return Err(GraphError::AnotherOpProducesOutput { op_id });
                }
                self.operation_nodes.push(node);
                Ok(node_id)
            }
            /// Given a `GraphState`, finds the next operation to perform in the graph
            ///
            /// # Panics
            pub fn get_next_operation<T: Clone + Debug>(
                &self,
                state: &GraphState<T>,
            ) -> (OperationId, Option<usize>) {
                let possible_ops = state.alive_representations.iter().flat_map(|(repr_id, _)| {
                    self.get_representation_input_op_ids(repr_id)
                        .into_iter()
                        .map(|op_id| (op_id, *repr_id))
                });
                let mut sorted_ops = possible_ops.collect::<Vec<_>>();
                sorted_ops.sort_by(|a, b| a.0.cmp(&b.0));
                let &(op, repr_id) = sorted_ops.first().unwrap();
                (op, repr_id.operation_step)
            }
            ///  Gets the reference counts of the `input_ids`
            ///
            /// # Description
            ///
            /// Calculates the number of operations that the inputs feed in the subgraph that produces the
            /// outputs. This includes the trivial operation if an output id = input id.
            ///
            /// This operation is expensive as it needs to calculate the subgraph and should be run as few
            /// times as necessary.
            ///
            /// # Panics
            pub fn get_reference_counts(
                &self,
                output_ids: &[RepresentationId],
                input_ids: &[RepresentationId],
            ) -> HashMap<RepresentationId, usize> {
                let mut reference_counts: HashMap<RepresentationId, usize> = input_ids
                    .iter()
                    .map(|&repr_id| (repr_id, 0_usize))
                    .collect();
                let operation_set = self.get_operation_set(output_ids, input_ids).unwrap();
                for out_id in output_ids {
                    if let Some(ref_count) = reference_counts.get_mut(out_id) {
                        *ref_count += 1;
                    }
                }
                for op_id in &operation_set {
                    let op_node = self.get_operation_node(op_id).unwrap();
                    let mut step_inputs_found = false;
                    op_node.get_output_ids().iter().for_each(|out_id| {
                        let mut step_ids = input_ids
                            .iter()
                            .filter(|&repr_id| {
                                repr_id.representation_node_id == out_id.representation_node_id
                            })
                            .collect::<Vec<_>>();
                        if !step_ids.is_empty() {
                            step_inputs_found = true;
                            step_ids.sort_by(|a, b| {
                                a.operation_step.unwrap().cmp(&b.operation_step.unwrap())
                            });
                            *reference_counts.get_mut(*step_ids.last().unwrap()).unwrap() += 1;
                        }
                    });
                    if step_inputs_found {
                        continue;
                    }
                    op_node.get_input_ids().iter().for_each(|in_id| {
                        if let Some(ref_count) = reference_counts.get_mut(in_id) {
                            *ref_count += 1;
                        }
                    });
                }
                reference_counts
            }
            /// Calculates a subgraph of operations necessary to compute the outputs from the inputs
            ///
            /// # Errors
            pub fn get_operation_set(
                &self,
                output_ids: &[RepresentationId],
                input_ids: &[RepresentationId],
            ) -> Result<HashSet<OperationId>, GraphError> {
                let mut active_representation_ids = output_ids.iter().collect::<Vec<_>>();
                let mut op_node_set: HashSet<OperationId> = HashSet::new();
                let mut finished_representations: HashSet<RepresentationId> =
                    input_ids.iter().copied().collect();
                while !active_representation_ids.is_empty() {
                    let active_repr_id = active_representation_ids
                        .pop()
                        .ok_or(GraphError::PoppedEmptyStack)?;
                    if finished_representations.contains(active_repr_id) {
                        continue;
                    }
                    let op_id = self.get_representation_op_id(active_repr_id).ok_or(
                        GraphError::NoOpCreatesRepresentation {
                            repr_id: *active_repr_id,
                        },
                    )?;
                    let op_node = self
                        .get_operation_node(&op_id)
                        .ok_or(GraphError::OperationNotExist { op_id })?;
                    op_node_set.insert(op_id);
                    if op_node
                        .get_output_ids()
                        .iter()
                        .map(|x| finished_representations.insert(*x))
                        .any(|x| !x)
                    {
                        return Err(GraphError::AnotherOpProducesOutput { op_id });
                    }
                    op_node.get_input_ids().iter().for_each(|input_id| {
                        if !finished_representations.contains(input_id) {
                            active_representation_ids.push(input_id);
                        }
                    });
                }
                Ok(op_node_set)
            }
        }
        impl fmt::Debug for Graph {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                let repr_ops_str = self
                    .representation_ops
                    .iter()
                    .map(|entry| {
                        let res = ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                            &["\t"],
                            &[::core::fmt::ArgumentV1::new_debug(&entry)],
                        ));
                        res
                    })
                    .join("\n");
                let op_strs = self
                    .operation_nodes
                    .iter()
                    .map(|node| {
                        let res = ::alloc::fmt::format(::core::fmt::Arguments::new_v1(
                            &["\t"],
                            &[::core::fmt::ArgumentV1::new_debug(&node)],
                        ));
                        res
                    })
                    .join("\n");
                f.write_fmt(::core::fmt::Arguments::new_v1(
                    &[
                        "Graph {\nrepresentation_ops: \n",
                        "\noperation_nodes: \n",
                        "\n}",
                    ],
                    &[
                        ::core::fmt::ArgumentV1::new_display(&&repr_ops_str),
                        ::core::fmt::ArgumentV1::new_display(&&op_strs),
                    ],
                ))
            }
        }
        pub enum GraphError {
            GenericError,
            PoppedEmptyStack,
            OperationNotExist { op_id: OperationId },
            NoOpCreatesRepresentation { repr_id: RepresentationId },
            AnotherOpProducesOutput { op_id: OperationId },
        }
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl ::core::fmt::Debug for GraphError {
            fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                match (&*self,) {
                    (&GraphError::GenericError,) => {
                        ::core::fmt::Formatter::write_str(f, "GenericError")
                    }
                    (&GraphError::PoppedEmptyStack,) => {
                        ::core::fmt::Formatter::write_str(f, "PoppedEmptyStack")
                    }
                    (&GraphError::OperationNotExist {
                        op_id: ref __self_0,
                    },) => {
                        let debug_trait_builder =
                            &mut ::core::fmt::Formatter::debug_struct(f, "OperationNotExist");
                        let _ = ::core::fmt::DebugStruct::field(
                            debug_trait_builder,
                            "op_id",
                            &&(*__self_0),
                        );
                        ::core::fmt::DebugStruct::finish(debug_trait_builder)
                    }
                    (&GraphError::NoOpCreatesRepresentation {
                        repr_id: ref __self_0,
                    },) => {
                        let debug_trait_builder = &mut ::core::fmt::Formatter::debug_struct(
                            f,
                            "NoOpCreatesRepresentation",
                        );
                        let _ = ::core::fmt::DebugStruct::field(
                            debug_trait_builder,
                            "repr_id",
                            &&(*__self_0),
                        );
                        ::core::fmt::DebugStruct::finish(debug_trait_builder)
                    }
                    (&GraphError::AnotherOpProducesOutput {
                        op_id: ref __self_0,
                    },) => {
                        let debug_trait_builder =
                            &mut ::core::fmt::Formatter::debug_struct(f, "AnotherOpProducesOutput");
                        let _ = ::core::fmt::DebugStruct::field(
                            debug_trait_builder,
                            "op_id",
                            &&(*__self_0),
                        );
                        ::core::fmt::DebugStruct::finish(debug_trait_builder)
                    }
                }
            }
        }
        pub enum PhysicalOp {
            Dense(Dense),
            Conv(Conv),
            ReLU(ReLU),
            Interpolate(Interpolate),
        }
        impl Debug for PhysicalOp {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                match self {
                    Self::Dense(x) => f.write_fmt(::core::fmt::Arguments::new_v1(
                        &[""],
                        &[::core::fmt::ArgumentV1::new_debug(&x)],
                    )),
                    Self::Conv(x) => f.write_fmt(::core::fmt::Arguments::new_v1(
                        &[""],
                        &[::core::fmt::ArgumentV1::new_debug(&x)],
                    )),
                    Self::ReLU(x) => f.write_fmt(::core::fmt::Arguments::new_v1(
                        &[""],
                        &[::core::fmt::ArgumentV1::new_debug(&x)],
                    )),
                    Self::Interpolate(x) => f.write_fmt(::core::fmt::Arguments::new_v1(
                        &[""],
                        &[::core::fmt::ArgumentV1::new_debug(&x)],
                    )),
                }
            }
        }
        impl Clone for PhysicalOp {
            fn clone(&self) -> Self {
                match self {
                    Self::Dense(x) => Self::Dense(x.clone()),
                    Self::Conv(x) => Self::Conv(x.clone()),
                    Self::ReLU(x) => Self::ReLU(x.clone()),
                    Self::Interpolate(x) => Self::Interpolate(x.clone()),
                }
            }
        }
        /// Each `RepresentationId` is created uniquely by a single `OperationNode`
        pub struct OperationNode {
            operation: PhysicalOp,
            inputs: Vec<RepresentationId>,
            outputs: Vec<RepresentationId>,
        }
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl ::core::fmt::Debug for OperationNode {
            fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                match *self {
                    OperationNode {
                        operation: ref __self_0_0,
                        inputs: ref __self_0_1,
                        outputs: ref __self_0_2,
                    } => {
                        let debug_trait_builder =
                            &mut ::core::fmt::Formatter::debug_struct(f, "OperationNode");
                        let _ = ::core::fmt::DebugStruct::field(
                            debug_trait_builder,
                            "operation",
                            &&(*__self_0_0),
                        );
                        let _ = ::core::fmt::DebugStruct::field(
                            debug_trait_builder,
                            "inputs",
                            &&(*__self_0_1),
                        );
                        let _ = ::core::fmt::DebugStruct::field(
                            debug_trait_builder,
                            "outputs",
                            &&(*__self_0_2),
                        );
                        ::core::fmt::DebugStruct::finish(debug_trait_builder)
                    }
                }
            }
        }
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl ::core::clone::Clone for OperationNode {
            #[inline]
            fn clone(&self) -> OperationNode {
                match *self {
                    OperationNode {
                        operation: ref __self_0_0,
                        inputs: ref __self_0_1,
                        outputs: ref __self_0_2,
                    } => OperationNode {
                        operation: ::core::clone::Clone::clone(&(*__self_0_0)),
                        inputs: ::core::clone::Clone::clone(&(*__self_0_1)),
                        outputs: ::core::clone::Clone::clone(&(*__self_0_2)),
                    },
                }
            }
        }
        impl OperationNode {
            pub fn new(
                operation: PhysicalOp,
                inputs: Vec<RepresentationId>,
                outputs: Vec<RepresentationId>,
            ) -> Self {
                Self {
                    operation,
                    inputs,
                    outputs,
                }
            }
            pub fn get_operation(&self) -> &PhysicalOp {
                &self.operation
            }
            pub const fn get_input_ids(&self) -> &Vec<RepresentationId> {
                &self.inputs
            }
            pub const fn get_output_ids(&self) -> &Vec<RepresentationId> {
                &self.outputs
            }
        }
    }
    mod operation {
        use crate::affine::Affine2;
        use crate::bounds::Bounds1;
        use crate::star::Star2;
        use crate::dnn::{Conv, Dense, Interpolate, ReLU};
        use crate::graph::PhysicalOp;
        use crate::tensorshape::TensorShape;
        use crate::NNVFloat;
        use enum_dispatch::enum_dispatch;
        use ndarray::{Array1, Array2};
        use std::any::Any;
        use std::fmt::{Debug, Display};
        use std::ops::Deref;
        /// Operations may not be stateful. I.e., they must deterministically produce identical outputs from identical inputs.
        /// State may be simulated with additional inputs/outputs and with a steppable operation. Further, the number of outputs
        /// from a step operation must be equal to the number of outputs from the non-stepped version of the operation.
        pub trait Operation: Clone + Debug + Send + Sync {
            fn as_any(&self) -> &dyn Any;
            fn num_steps(&self) -> Option<usize> {
                None
            }
            fn input_shapes(&self) -> Vec<TensorShape> {
                ::core::panicking::panic("explicit panic")
            }
            fn output_shapes(&self) -> Vec<TensorShape> {
                ::core::panicking::panic("explicit panic")
            }
            fn forward1(&self, input: &[&Array1<NNVFloat>]) -> Vec<Array1<NNVFloat>>;
            fn forward2(&self, input: &[&Array2<NNVFloat>]) -> Vec<Array2<NNVFloat>>;
            fn apply_bounds(
                &self,
                bounds: &[Bounds1],
                lower_aff: &[Affine2],
                upper_aff: &[Affine2],
            ) -> Vec<(Bounds1, Affine2, Affine2)>;
            fn apply_bounds_step(
                &self,
                _dim: usize,
                _bounds: &[Bounds1],
                _lower_aff: &[Affine2],
                _upper_aff: &[Affine2],
            ) -> Vec<(Bounds1, Affine2, Affine2)> {
                ::core::panicking::panic("explicit panic");
            }
            /// Returns the set of children stars with their input_bounds.
            /// In the case that there is one, sets the bool to whether the output bounds can be copied.
            ///
            /// We pass axis_aligned_input_bounds through each operation because it's very cheap to update and expensive to calculate.
            ///
            /// # Arguments
            ///
            /// * `parent_stars` - The stars used as input to the operation.
            /// * `step_id` - The (optional) step of the operation.
            /// * `axis_aligned_input_bounds` - Optional outer bounds on the entire DNN's input set, must be passed if it's defined on the StarSet
            ///
            /// # Returns
            ///
            /// * `child_stars` -
            /// * `children_axis_aligned_input_bounds` -
            /// * `same_output_bounds` - Whether the children have the same output bounds as the parents. See assumptions above.
            fn forward_star<StarRef: Deref<Target = Star2>>(
                &self,
                parent_stars: Vec<StarRef>,
                step_id: Option<usize>,
                parent_axis_aligned_input_bounds: Vec<&Bounds1>,
            ) -> (Vec<Star2>, Vec<Bounds1>, bool);
            fn inputs_dims(&self) -> Vec<usize> {
                self.input_shapes()
                    .into_iter()
                    .map(|input| input.dims())
                    .collect()
            }
            fn outputs_dims(&self) -> Vec<usize> {
                self.input_shapes()
                    .into_iter()
                    .map(|output| output.dims())
                    .collect()
            }
            fn is_activation(&self) -> bool {
                false
            }
            fn get_activation_pattern(
                &self,
                _state: &[&Array2<NNVFloat>],
            ) -> Option<Vec<Array2<bool>>> {
                None
            }
        }
        impl ::core::convert::From<Dense> for PhysicalOp {
            fn from(v: Dense) -> PhysicalOp {
                PhysicalOp::Dense(v)
            }
        }
        impl ::core::convert::From<Conv> for PhysicalOp {
            fn from(v: Conv) -> PhysicalOp {
                PhysicalOp::Conv(v)
            }
        }
        impl ::core::convert::From<ReLU> for PhysicalOp {
            fn from(v: ReLU) -> PhysicalOp {
                PhysicalOp::ReLU(v)
            }
        }
        impl ::core::convert::From<Interpolate> for PhysicalOp {
            fn from(v: Interpolate) -> PhysicalOp {
                PhysicalOp::Interpolate(v)
            }
        }
        impl core::convert::TryInto<Dense> for PhysicalOp {
            type Error = &'static str;
            fn try_into(
                self,
            ) -> ::core::result::Result<Dense, <Self as core::convert::TryInto<Dense>>::Error>
            {
                match self {
                    PhysicalOp::Dense(v) => Ok(v),
                    PhysicalOp::Conv(v) => Err("Tried to convert variant Conv to Dense"),
                    PhysicalOp::ReLU(v) => Err("Tried to convert variant ReLU to Dense"),
                    PhysicalOp::Interpolate(v) => {
                        Err("Tried to convert variant Interpolate to Dense")
                    }
                }
            }
        }
        impl core::convert::TryInto<Conv> for PhysicalOp {
            type Error = &'static str;
            fn try_into(
                self,
            ) -> ::core::result::Result<Conv, <Self as core::convert::TryInto<Conv>>::Error>
            {
                match self {
                    PhysicalOp::Conv(v) => Ok(v),
                    PhysicalOp::Dense(v) => Err("Tried to convert variant Dense to Conv"),
                    PhysicalOp::ReLU(v) => Err("Tried to convert variant ReLU to Conv"),
                    PhysicalOp::Interpolate(v) => {
                        Err("Tried to convert variant Interpolate to Conv")
                    }
                }
            }
        }
        impl core::convert::TryInto<ReLU> for PhysicalOp {
            type Error = &'static str;
            fn try_into(
                self,
            ) -> ::core::result::Result<ReLU, <Self as core::convert::TryInto<ReLU>>::Error>
            {
                match self {
                    PhysicalOp::ReLU(v) => Ok(v),
                    PhysicalOp::Dense(v) => Err("Tried to convert variant Dense to ReLU"),
                    PhysicalOp::Conv(v) => Err("Tried to convert variant Conv to ReLU"),
                    PhysicalOp::Interpolate(v) => {
                        Err("Tried to convert variant Interpolate to ReLU")
                    }
                }
            }
        }
        impl core::convert::TryInto<Interpolate> for PhysicalOp {
            type Error = &'static str;
            fn try_into(
                self,
            ) -> ::core::result::Result<
                Interpolate,
                <Self as core::convert::TryInto<Interpolate>>::Error,
            > {
                match self {
                    PhysicalOp::Interpolate(v) => Ok(v),
                    PhysicalOp::Dense(v) => Err("Tried to convert variant Dense to Interpolate"),
                    PhysicalOp::Conv(v) => Err("Tried to convert variant Conv to Interpolate"),
                    PhysicalOp::ReLU(v) => Err("Tried to convert variant ReLU to Interpolate"),
                }
            }
        }
        impl Operation for PhysicalOp {
            #[inline]
            fn as_any(&self) -> &dyn Any {
                match self {
                    PhysicalOp::Dense(inner) => Operation::as_any(inner),
                    PhysicalOp::Conv(inner) => Operation::as_any(inner),
                    PhysicalOp::ReLU(inner) => Operation::as_any(inner),
                    PhysicalOp::Interpolate(inner) => Operation::as_any(inner),
                }
            }
            #[inline]
            fn num_steps(&self) -> Option<usize> {
                match self {
                    PhysicalOp::Dense(inner) => Operation::num_steps(inner),
                    PhysicalOp::Conv(inner) => Operation::num_steps(inner),
                    PhysicalOp::ReLU(inner) => Operation::num_steps(inner),
                    PhysicalOp::Interpolate(inner) => Operation::num_steps(inner),
                }
            }
            #[inline]
            fn input_shapes(&self) -> Vec<TensorShape> {
                match self {
                    PhysicalOp::Dense(inner) => Operation::input_shapes(inner),
                    PhysicalOp::Conv(inner) => Operation::input_shapes(inner),
                    PhysicalOp::ReLU(inner) => Operation::input_shapes(inner),
                    PhysicalOp::Interpolate(inner) => Operation::input_shapes(inner),
                }
            }
            #[inline]
            fn output_shapes(&self) -> Vec<TensorShape> {
                match self {
                    PhysicalOp::Dense(inner) => Operation::output_shapes(inner),
                    PhysicalOp::Conv(inner) => Operation::output_shapes(inner),
                    PhysicalOp::ReLU(inner) => Operation::output_shapes(inner),
                    PhysicalOp::Interpolate(inner) => Operation::output_shapes(inner),
                }
            }
            #[inline]
            fn forward1(
                &self,
                __enum_dispatch_arg_0: &[&Array1<NNVFloat>],
            ) -> Vec<Array1<NNVFloat>> {
                match self {
                    PhysicalOp::Dense(inner) => Operation::forward1(inner, __enum_dispatch_arg_0),
                    PhysicalOp::Conv(inner) => Operation::forward1(inner, __enum_dispatch_arg_0),
                    PhysicalOp::ReLU(inner) => Operation::forward1(inner, __enum_dispatch_arg_0),
                    PhysicalOp::Interpolate(inner) => {
                        Operation::forward1(inner, __enum_dispatch_arg_0)
                    }
                }
            }
            #[inline]
            fn forward2(
                &self,
                __enum_dispatch_arg_0: &[&Array2<NNVFloat>],
            ) -> Vec<Array2<NNVFloat>> {
                match self {
                    PhysicalOp::Dense(inner) => Operation::forward2(inner, __enum_dispatch_arg_0),
                    PhysicalOp::Conv(inner) => Operation::forward2(inner, __enum_dispatch_arg_0),
                    PhysicalOp::ReLU(inner) => Operation::forward2(inner, __enum_dispatch_arg_0),
                    PhysicalOp::Interpolate(inner) => {
                        Operation::forward2(inner, __enum_dispatch_arg_0)
                    }
                }
            }
            #[inline]
            fn apply_bounds(
                &self,
                __enum_dispatch_arg_0: &[Bounds1],
                __enum_dispatch_arg_1: &[Affine2],
                __enum_dispatch_arg_2: &[Affine2],
            ) -> Vec<(Bounds1, Affine2, Affine2)> {
                match self {
                    PhysicalOp::Dense(inner) => Operation::apply_bounds(
                        inner,
                        __enum_dispatch_arg_0,
                        __enum_dispatch_arg_1,
                        __enum_dispatch_arg_2,
                    ),
                    PhysicalOp::Conv(inner) => Operation::apply_bounds(
                        inner,
                        __enum_dispatch_arg_0,
                        __enum_dispatch_arg_1,
                        __enum_dispatch_arg_2,
                    ),
                    PhysicalOp::ReLU(inner) => Operation::apply_bounds(
                        inner,
                        __enum_dispatch_arg_0,
                        __enum_dispatch_arg_1,
                        __enum_dispatch_arg_2,
                    ),
                    PhysicalOp::Interpolate(inner) => Operation::apply_bounds(
                        inner,
                        __enum_dispatch_arg_0,
                        __enum_dispatch_arg_1,
                        __enum_dispatch_arg_2,
                    ),
                }
            }
            #[inline]
            fn apply_bounds_step(
                &self,
                __enum_dispatch_arg_0: usize,
                __enum_dispatch_arg_1: &[Bounds1],
                __enum_dispatch_arg_2: &[Affine2],
                __enum_dispatch_arg_3: &[Affine2],
            ) -> Vec<(Bounds1, Affine2, Affine2)> {
                match self {
                    PhysicalOp::Dense(inner) => Operation::apply_bounds_step(
                        inner,
                        __enum_dispatch_arg_0,
                        __enum_dispatch_arg_1,
                        __enum_dispatch_arg_2,
                        __enum_dispatch_arg_3,
                    ),
                    PhysicalOp::Conv(inner) => Operation::apply_bounds_step(
                        inner,
                        __enum_dispatch_arg_0,
                        __enum_dispatch_arg_1,
                        __enum_dispatch_arg_2,
                        __enum_dispatch_arg_3,
                    ),
                    PhysicalOp::ReLU(inner) => Operation::apply_bounds_step(
                        inner,
                        __enum_dispatch_arg_0,
                        __enum_dispatch_arg_1,
                        __enum_dispatch_arg_2,
                        __enum_dispatch_arg_3,
                    ),
                    PhysicalOp::Interpolate(inner) => Operation::apply_bounds_step(
                        inner,
                        __enum_dispatch_arg_0,
                        __enum_dispatch_arg_1,
                        __enum_dispatch_arg_2,
                        __enum_dispatch_arg_3,
                    ),
                }
            }
            /// Returns the set of children stars with their input_bounds.
            /// In the case that there is one, sets the bool to whether the output bounds can be copied.
            ///
            /// We pass axis_aligned_input_bounds through each operation because it's very cheap to update and expensive to calculate.
            ///
            /// # Arguments
            ///
            /// * `parent_stars` - The stars used as input to the operation.
            /// * `step_id` - The (optional) step of the operation.
            /// * `axis_aligned_input_bounds` - Optional outer bounds on the entire DNN's input set, must be passed if it's defined on the StarSet
            ///
            /// # Returns
            ///
            /// * `child_stars` -
            /// * `children_axis_aligned_input_bounds` -
            /// * `same_output_bounds` - Whether the children have the same output bounds as the parents. See assumptions above.
            #[inline]
            fn forward_star<StarRef: Deref<Target = Star2>>(
                &self,
                __enum_dispatch_arg_0: Vec<StarRef>,
                __enum_dispatch_arg_1: Option<usize>,
                __enum_dispatch_arg_2: Vec<&Bounds1>,
            ) -> (Vec<Star2>, Vec<Bounds1>, bool) {
                match self {
                    PhysicalOp::Dense(inner) => Operation::forward_star::<StarRef>(
                        inner,
                        __enum_dispatch_arg_0,
                        __enum_dispatch_arg_1,
                        __enum_dispatch_arg_2,
                    ),
                    PhysicalOp::Conv(inner) => Operation::forward_star::<StarRef>(
                        inner,
                        __enum_dispatch_arg_0,
                        __enum_dispatch_arg_1,
                        __enum_dispatch_arg_2,
                    ),
                    PhysicalOp::ReLU(inner) => Operation::forward_star::<StarRef>(
                        inner,
                        __enum_dispatch_arg_0,
                        __enum_dispatch_arg_1,
                        __enum_dispatch_arg_2,
                    ),
                    PhysicalOp::Interpolate(inner) => Operation::forward_star::<StarRef>(
                        inner,
                        __enum_dispatch_arg_0,
                        __enum_dispatch_arg_1,
                        __enum_dispatch_arg_2,
                    ),
                }
            }
            #[inline]
            fn inputs_dims(&self) -> Vec<usize> {
                match self {
                    PhysicalOp::Dense(inner) => Operation::inputs_dims(inner),
                    PhysicalOp::Conv(inner) => Operation::inputs_dims(inner),
                    PhysicalOp::ReLU(inner) => Operation::inputs_dims(inner),
                    PhysicalOp::Interpolate(inner) => Operation::inputs_dims(inner),
                }
            }
            #[inline]
            fn outputs_dims(&self) -> Vec<usize> {
                match self {
                    PhysicalOp::Dense(inner) => Operation::outputs_dims(inner),
                    PhysicalOp::Conv(inner) => Operation::outputs_dims(inner),
                    PhysicalOp::ReLU(inner) => Operation::outputs_dims(inner),
                    PhysicalOp::Interpolate(inner) => Operation::outputs_dims(inner),
                }
            }
            #[inline]
            fn is_activation(&self) -> bool {
                match self {
                    PhysicalOp::Dense(inner) => Operation::is_activation(inner),
                    PhysicalOp::Conv(inner) => Operation::is_activation(inner),
                    PhysicalOp::ReLU(inner) => Operation::is_activation(inner),
                    PhysicalOp::Interpolate(inner) => Operation::is_activation(inner),
                }
            }
            #[inline]
            fn get_activation_pattern(
                &self,
                __enum_dispatch_arg_0: &[&Array2<NNVFloat>],
            ) -> Option<Vec<Array2<bool>>> {
                match self {
                    PhysicalOp::Dense(inner) => {
                        Operation::get_activation_pattern(inner, __enum_dispatch_arg_0)
                    }
                    PhysicalOp::Conv(inner) => {
                        Operation::get_activation_pattern(inner, __enum_dispatch_arg_0)
                    }
                    PhysicalOp::ReLU(inner) => {
                        Operation::get_activation_pattern(inner, __enum_dispatch_arg_0)
                    }
                    PhysicalOp::Interpolate(inner) => {
                        Operation::get_activation_pattern(inner, __enum_dispatch_arg_0)
                    }
                }
            }
        }
    }
    pub use execute_engine::{Engine, ExecuteError};
    pub use graph::{
        Graph, GraphError, GraphState, OperationId, OperationNode, PhysicalOp, RepresentationId,
    };
    pub(crate) use operation::Operation;
}
pub mod lp {
    use ndarray::Array1;
    pub enum LinearSolution {
        Solution(Array1<f64>, f64),
        Infeasible,
        Unbounded(Array1<f64>),
    }
    #[automatically_derived]
    #[allow(unused_qualifications)]
    impl ::core::fmt::Debug for LinearSolution {
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            match (&*self,) {
                (&LinearSolution::Solution(ref __self_0, ref __self_1),) => {
                    let debug_trait_builder =
                        &mut ::core::fmt::Formatter::debug_tuple(f, "Solution");
                    let _ = ::core::fmt::DebugTuple::field(debug_trait_builder, &&(*__self_0));
                    let _ = ::core::fmt::DebugTuple::field(debug_trait_builder, &&(*__self_1));
                    ::core::fmt::DebugTuple::finish(debug_trait_builder)
                }
                (&LinearSolution::Infeasible,) => {
                    ::core::fmt::Formatter::write_str(f, "Infeasible")
                }
                (&LinearSolution::Unbounded(ref __self_0),) => {
                    let debug_trait_builder =
                        &mut ::core::fmt::Formatter::debug_tuple(f, "Unbounded");
                    let _ = ::core::fmt::DebugTuple::field(debug_trait_builder, &&(*__self_0));
                    ::core::fmt::DebugTuple::finish(debug_trait_builder)
                }
            }
        }
    }
}
pub mod polytope {
    use crate::affine::Affine2;
    use crate::bounds::Bounds1;
    use crate::gaussian::GaussianDistribution;
    use crate::lp::solve;
    use crate::lp::LinearSolution;
    use crate::ndarray_linalg::SVD;
    use crate::NNVFloat;
    use ndarray::arr1;
    use ndarray::array;
    use ndarray::concatenate;
    use ndarray::Axis;
    use ndarray::Ix2;
    use ndarray::Slice;
    use ndarray::Zip;
    use ndarray::{Array1, Array2};
    use ndarray::{ArrayView1, ArrayView2, ArrayViewMut1};
    use ndarray_stats::QuantileExt;
    use num::Zero;
    use serde::{Deserialize, Serialize};
    use std::convert::TryFrom;
    use std::iter;
    use std::ops::Mul;
    use std::ops::MulAssign;
    use truncnorm::distributions::MultivariateTruncatedNormal;
    pub struct Polytope {
        coeffs: Array2<NNVFloat>,
        rhs: Array1<NNVFloat>,
    }
    #[automatically_derived]
    #[allow(unused_qualifications)]
    impl ::core::clone::Clone for Polytope {
        #[inline]
        fn clone(&self) -> Polytope {
            match *self {
                Polytope {
                    coeffs: ref __self_0_0,
                    rhs: ref __self_0_1,
                } => Polytope {
                    coeffs: ::core::clone::Clone::clone(&(*__self_0_0)),
                    rhs: ::core::clone::Clone::clone(&(*__self_0_1)),
                },
            }
        }
    }
    #[automatically_derived]
    #[allow(unused_qualifications)]
    impl ::core::fmt::Debug for Polytope {
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            match *self {
                Polytope {
                    coeffs: ref __self_0_0,
                    rhs: ref __self_0_1,
                } => {
                    let debug_trait_builder =
                        &mut ::core::fmt::Formatter::debug_struct(f, "Polytope");
                    let _ = ::core::fmt::DebugStruct::field(
                        debug_trait_builder,
                        "coeffs",
                        &&(*__self_0_0),
                    );
                    let _ = ::core::fmt::DebugStruct::field(
                        debug_trait_builder,
                        "rhs",
                        &&(*__self_0_1),
                    );
                    ::core::fmt::DebugStruct::finish(debug_trait_builder)
                }
            }
        }
    }
    #[doc(hidden)]
    #[allow(non_upper_case_globals, unused_attributes, unused_qualifications)]
    const _: () = {
        #[allow(unused_extern_crates, clippy::useless_attribute)]
        extern crate serde as _serde;
        #[automatically_derived]
        impl<'de> _serde::Deserialize<'de> for Polytope {
            fn deserialize<__D>(__deserializer: __D) -> _serde::__private::Result<Self, __D::Error>
            where
                __D: _serde::Deserializer<'de>,
            {
                #[allow(non_camel_case_types)]
                enum __Field {
                    __field0,
                    __field1,
                    __ignore,
                }
                struct __FieldVisitor;
                impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                    type Value = __Field;
                    fn expecting(
                        &self,
                        __formatter: &mut _serde::__private::Formatter,
                    ) -> _serde::__private::fmt::Result {
                        _serde::__private::Formatter::write_str(__formatter, "field identifier")
                    }
                    fn visit_u64<__E>(
                        self,
                        __value: u64,
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            0u64 => _serde::__private::Ok(__Field::__field0),
                            1u64 => _serde::__private::Ok(__Field::__field1),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                    fn visit_str<__E>(
                        self,
                        __value: &str,
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            "coeffs" => _serde::__private::Ok(__Field::__field0),
                            "rhs" => _serde::__private::Ok(__Field::__field1),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                    fn visit_bytes<__E>(
                        self,
                        __value: &[u8],
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            b"coeffs" => _serde::__private::Ok(__Field::__field0),
                            b"rhs" => _serde::__private::Ok(__Field::__field1),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                }
                impl<'de> _serde::Deserialize<'de> for __Field {
                    #[inline]
                    fn deserialize<__D>(
                        __deserializer: __D,
                    ) -> _serde::__private::Result<Self, __D::Error>
                    where
                        __D: _serde::Deserializer<'de>,
                    {
                        _serde::Deserializer::deserialize_identifier(__deserializer, __FieldVisitor)
                    }
                }
                struct __Visitor<'de> {
                    marker: _serde::__private::PhantomData<Polytope>,
                    lifetime: _serde::__private::PhantomData<&'de ()>,
                }
                impl<'de> _serde::de::Visitor<'de> for __Visitor<'de> {
                    type Value = Polytope;
                    fn expecting(
                        &self,
                        __formatter: &mut _serde::__private::Formatter,
                    ) -> _serde::__private::fmt::Result {
                        _serde::__private::Formatter::write_str(__formatter, "struct Polytope")
                    }
                    #[inline]
                    fn visit_seq<__A>(
                        self,
                        mut __seq: __A,
                    ) -> _serde::__private::Result<Self::Value, __A::Error>
                    where
                        __A: _serde::de::SeqAccess<'de>,
                    {
                        let __field0 = match match _serde::de::SeqAccess::next_element::<
                            Array2<NNVFloat>,
                        >(&mut __seq)
                        {
                            _serde::__private::Ok(__val) => __val,
                            _serde::__private::Err(__err) => {
                                return _serde::__private::Err(__err);
                            }
                        } {
                            _serde::__private::Some(__value) => __value,
                            _serde::__private::None => {
                                return _serde::__private::Err(_serde::de::Error::invalid_length(
                                    0usize,
                                    &"struct Polytope with 2 elements",
                                ));
                            }
                        };
                        let __field1 = match match _serde::de::SeqAccess::next_element::<
                            Array1<NNVFloat>,
                        >(&mut __seq)
                        {
                            _serde::__private::Ok(__val) => __val,
                            _serde::__private::Err(__err) => {
                                return _serde::__private::Err(__err);
                            }
                        } {
                            _serde::__private::Some(__value) => __value,
                            _serde::__private::None => {
                                return _serde::__private::Err(_serde::de::Error::invalid_length(
                                    1usize,
                                    &"struct Polytope with 2 elements",
                                ));
                            }
                        };
                        _serde::__private::Ok(Polytope {
                            coeffs: __field0,
                            rhs: __field1,
                        })
                    }
                    #[inline]
                    fn visit_map<__A>(
                        self,
                        mut __map: __A,
                    ) -> _serde::__private::Result<Self::Value, __A::Error>
                    where
                        __A: _serde::de::MapAccess<'de>,
                    {
                        let mut __field0: _serde::__private::Option<Array2<NNVFloat>> =
                            _serde::__private::None;
                        let mut __field1: _serde::__private::Option<Array1<NNVFloat>> =
                            _serde::__private::None;
                        while let _serde::__private::Some(__key) =
                            match _serde::de::MapAccess::next_key::<__Field>(&mut __map) {
                                _serde::__private::Ok(__val) => __val,
                                _serde::__private::Err(__err) => {
                                    return _serde::__private::Err(__err);
                                }
                            }
                        {
                            match __key {
                                __Field::__field0 => {
                                    if _serde::__private::Option::is_some(&__field0) {
                                        return _serde::__private::Err(
                                            <__A::Error as _serde::de::Error>::duplicate_field(
                                                "coeffs",
                                            ),
                                        );
                                    }
                                    __field0 = _serde::__private::Some(
                                        match _serde::de::MapAccess::next_value::<Array2<NNVFloat>>(
                                            &mut __map,
                                        ) {
                                            _serde::__private::Ok(__val) => __val,
                                            _serde::__private::Err(__err) => {
                                                return _serde::__private::Err(__err);
                                            }
                                        },
                                    );
                                }
                                __Field::__field1 => {
                                    if _serde::__private::Option::is_some(&__field1) {
                                        return _serde::__private::Err(
                                            <__A::Error as _serde::de::Error>::duplicate_field(
                                                "rhs",
                                            ),
                                        );
                                    }
                                    __field1 = _serde::__private::Some(
                                        match _serde::de::MapAccess::next_value::<Array1<NNVFloat>>(
                                            &mut __map,
                                        ) {
                                            _serde::__private::Ok(__val) => __val,
                                            _serde::__private::Err(__err) => {
                                                return _serde::__private::Err(__err);
                                            }
                                        },
                                    );
                                }
                                _ => {
                                    let _ = match _serde::de::MapAccess::next_value::<
                                        _serde::de::IgnoredAny,
                                    >(&mut __map)
                                    {
                                        _serde::__private::Ok(__val) => __val,
                                        _serde::__private::Err(__err) => {
                                            return _serde::__private::Err(__err);
                                        }
                                    };
                                }
                            }
                        }
                        let __field0 = match __field0 {
                            _serde::__private::Some(__field0) => __field0,
                            _serde::__private::None => {
                                match _serde::__private::de::missing_field("coeffs") {
                                    _serde::__private::Ok(__val) => __val,
                                    _serde::__private::Err(__err) => {
                                        return _serde::__private::Err(__err);
                                    }
                                }
                            }
                        };
                        let __field1 = match __field1 {
                            _serde::__private::Some(__field1) => __field1,
                            _serde::__private::None => {
                                match _serde::__private::de::missing_field("rhs") {
                                    _serde::__private::Ok(__val) => __val,
                                    _serde::__private::Err(__err) => {
                                        return _serde::__private::Err(__err);
                                    }
                                }
                            }
                        };
                        _serde::__private::Ok(Polytope {
                            coeffs: __field0,
                            rhs: __field1,
                        })
                    }
                }
                const FIELDS: &'static [&'static str] = &["coeffs", "rhs"];
                _serde::Deserializer::deserialize_struct(
                    __deserializer,
                    "Polytope",
                    FIELDS,
                    __Visitor {
                        marker: _serde::__private::PhantomData::<Polytope>,
                        lifetime: _serde::__private::PhantomData,
                    },
                )
            }
        }
    };
    #[doc(hidden)]
    #[allow(non_upper_case_globals, unused_attributes, unused_qualifications)]
    const _: () = {
        #[allow(unused_extern_crates, clippy::useless_attribute)]
        extern crate serde as _serde;
        #[automatically_derived]
        impl _serde::Serialize for Polytope {
            fn serialize<__S>(
                &self,
                __serializer: __S,
            ) -> _serde::__private::Result<__S::Ok, __S::Error>
            where
                __S: _serde::Serializer,
            {
                let mut __serde_state = match _serde::Serializer::serialize_struct(
                    __serializer,
                    "Polytope",
                    false as usize + 1 + 1,
                ) {
                    _serde::__private::Ok(__val) => __val,
                    _serde::__private::Err(__err) => {
                        return _serde::__private::Err(__err);
                    }
                };
                match _serde::ser::SerializeStruct::serialize_field(
                    &mut __serde_state,
                    "coeffs",
                    &self.coeffs,
                ) {
                    _serde::__private::Ok(__val) => __val,
                    _serde::__private::Err(__err) => {
                        return _serde::__private::Err(__err);
                    }
                };
                match _serde::ser::SerializeStruct::serialize_field(
                    &mut __serde_state,
                    "rhs",
                    &self.rhs,
                ) {
                    _serde::__private::Ok(__val) => __val,
                    _serde::__private::Err(__err) => {
                        return _serde::__private::Err(__err);
                    }
                };
                _serde::ser::SerializeStruct::end(__serde_state)
            }
        }
    };
    impl Polytope {
        /// # Panics
        pub fn nonempty_new(coeffs: &Array2<NNVFloat>, rhs: &Array1<NNVFloat>) -> Option<Self> {
            let (fcoeffs, frhs): (Vec<ArrayView1<NNVFloat>>, Vec<NNVFloat>) = coeffs
                .rows()
                .into_iter()
                .zip(rhs.iter())
                .filter(|(row, rhs)| row.iter().any(|x| x.abs() > 1e-15) || **rhs < 0.)
                .unzip();
            let fscoeffs: Vec<ArrayView2<NNVFloat>> = fcoeffs
                .into_iter()
                .map(|x| x.insert_axis(Axis(0)))
                .collect();
            if frhs.is_empty() {
                return None;
            }
            let coeffs = concatenate(Axis(0), &fscoeffs).unwrap();
            let rhs = Array1::from_vec(frhs);
            Some(Self::new(coeffs, rhs))
        }
        pub fn new(coeffs: Array2<NNVFloat>, rhs: Array1<NNVFloat>) -> Self {
            if true {
                match (&coeffs.nrows(), &rhs.len()) {
                    (left_val, right_val) => {
                        if !(*left_val == *right_val) {
                            let kind = ::core::panicking::AssertKind::Eq;
                            ::core::panicking::assert_failed(
                                kind,
                                &*left_val,
                                &*right_val,
                                ::core::option::Option::None,
                            );
                        }
                    }
                };
            };
            Self { coeffs, rhs }
        }
        pub fn coeffs(&self) -> ArrayView2<NNVFloat> {
            self.coeffs.view()
        }
        pub fn rhs(&self) -> ArrayView1<NNVFloat> {
            self.rhs.view()
        }
        pub fn rhs_mut(&mut self) -> ArrayViewMut1<NNVFloat> {
            self.rhs.view_mut()
        }
        pub fn num_dims(&self) -> usize {
            self.coeffs.ncols()
        }
        pub fn num_constraints(&self) -> usize {
            self.rhs.len()
        }
        #[must_use]
        pub fn intersect(&self, other: &Self) -> Self {
            Self {
                coeffs: ::ndarray::concatenate(
                    Axis(0),
                    &[
                        ::ndarray::ArrayView::from(&self.coeffs),
                        ::ndarray::ArrayView::from(&other.coeffs),
                    ],
                )
                .unwrap(),
                rhs: ::ndarray::concatenate(
                    Axis(0),
                    &[
                        ::ndarray::ArrayView::from(&self.rhs),
                        ::ndarray::ArrayView::from(&other.rhs),
                    ],
                )
                .unwrap(),
            }
        }
        pub fn check_redundant(
            &self,
            coeffs: ArrayView1<NNVFloat>,
            rhs: NNVFloat,
            bounds: &Option<Bounds1>,
        ) -> bool {
            let maximize_eqn = &coeffs * -1.;
            let maximize_rhs = rhs + 1.;
            let solved = solve(
                self.coeffs()
                    .rows()
                    .into_iter()
                    .chain(iter::once(maximize_eqn.view())),
                ::ndarray::concatenate(
                    Axis(0),
                    &[
                        ::ndarray::ArrayView::from(&self.rhs()),
                        ::ndarray::ArrayView::from(&arr1(&[maximize_rhs])),
                    ],
                )
                .unwrap()
                .view(),
                maximize_eqn.view(),
                bounds.as_ref(),
            );
            let val: f64 = match solved {
                LinearSolution::Solution(_, val) => val,
                LinearSolution::Infeasible | LinearSolution::Unbounded(_) => return true,
            };
            val > rhs
        }
        /// `check_redundant` is currently disabled
        /// # Panics
        pub fn add_eqn(&mut self, coeffs: ArrayView1<NNVFloat>, rhs: NNVFloat) {
            if coeffs.iter().all(|x| x.abs() < 1e-15) {
                return;
            }
            self.coeffs
                .append(Axis(0), coeffs.insert_axis(Axis(0)).view())
                .unwrap();
            self.rhs
                .append(
                    Axis(0),
                    { ::ndarray::Array::from(<[_]>::into_vec(box [rhs])) }.view(),
                )
                .unwrap();
        }
        pub fn remove_eqn(&mut self, idx: usize) {
            if true {
                if !(idx < self.coeffs().nrows()) {
                    ::core::panicking::panic("assertion failed: idx < self.coeffs().nrows()")
                };
            };
            self.coeffs.remove_index(Axis(0), idx);
            self.rhs.remove_index(Axis(0), idx);
        }
        pub fn any_nan(&self) -> bool {
            self.coeffs().iter().any(|x| x.is_nan()) || self.rhs.iter().any(|x| x.is_nan())
        }
        /// # Panics
        pub fn filter_trivial(&mut self) {
            let (coeffs, rhs): (Vec<ArrayView1<NNVFloat>>, Vec<_>) = self
                .coeffs
                .rows()
                .into_iter()
                .zip(self.rhs().iter())
                .filter(|(coeffs, _rhs)| !coeffs.iter().all(|x| *x == 0.))
                .unzip();
            self.coeffs = ndarray::stack(Axis(0), &coeffs).unwrap();
            self.rhs = Array1::from_vec(rhs);
        }
        /// # Panics
        #[must_use]
        pub fn get_eqn(&self, idx: usize) -> Self {
            let i_idx: isize = isize::try_from(idx).unwrap();
            Self {
                coeffs: self
                    .coeffs
                    .slice_axis(Axis(0), Slice::new(i_idx, Some(i_idx + 1), 1))
                    .to_owned(),
                rhs: self
                    .rhs
                    .slice_axis(Axis(0), Slice::new(i_idx, Some(i_idx + 1), 1))
                    .to_owned(),
            }
        }
        /// Returns whether a given point is in the set represented by the `Polytope`
        pub fn is_member(&self, point: &ArrayView1<NNVFloat>) -> bool {
            let vals = self.coeffs.dot(point);
            Zip::from(&self.rhs)
                .and(&vals)
                .fold(true, |acc, ub, v| acc && (v <= ub))
        }
        /// Remove dimensions from the Polytope that have fixed value.
        ///
        /// # Arguments
        ///
        /// * `x` - An Array with fixed values in each dimension. Any dimensions that aren't fixed should be set to zero.
        /// * `fixed_idxs` - Array that indicates which dimensions are fixed with `true` (because a dim could be fixed at zero)
        ///
        /// # Returns
        ///
        /// None if all the indices are fixed, otherwise a `Self` with reduced dimension
        ///
        /// # Panics
        ///
        /// Only if the underlying struct is malformed
        pub fn reduce_with_values(
            &self,
            x: ArrayView1<NNVFloat>,
            fixed_idxs: ArrayView1<bool>,
        ) -> Option<Self> {
            if fixed_idxs.iter().all(|y| *y) {
                return None;
            }
            let reduced_rhs: Array1<NNVFloat> = &self.rhs - self.coeffs.dot(&x);
            let filtered_coeffs = {
                let filtered_cols: Vec<ArrayView2<NNVFloat>> = self
                    .coeffs
                    .columns()
                    .into_iter()
                    .zip(fixed_idxs.iter())
                    .filter(|(_item, &is_fixed)| !is_fixed)
                    .map(|x| x.0.insert_axis(Axis(1)))
                    .collect();
                concatenate(Axis(1), &filtered_cols).unwrap()
            };
            let is_nontrivial: Vec<bool> = filtered_coeffs
                .rows()
                .clone()
                .into_iter()
                .map(|row| row.iter().all(|x| !x.is_zero()))
                .collect();
            if !is_nontrivial.iter().any(|y| *y) {
                return None;
            }
            let nontrivial_coeffs: Vec<ArrayView2<NNVFloat>> = filtered_coeffs
                .rows()
                .into_iter()
                .zip(is_nontrivial.iter())
                .filter(|(_, &is_nontrivial)| is_nontrivial)
                .map(|(row, _)| row.insert_axis(Axis(0)))
                .collect();
            let nontrivial_rhs: Vec<NNVFloat> = reduced_rhs
                .into_iter()
                .zip(is_nontrivial.iter())
                .filter(|(_, &is_nontrivial)| is_nontrivial)
                .map(|(val, _)| val)
                .collect();
            let final_coeffs: Array2<NNVFloat> = concatenate(Axis(0), &nontrivial_coeffs).unwrap();
            let final_rhs = Array1::from_vec(nontrivial_rhs);
            Some(Self::new(final_coeffs, final_rhs))
        }
        /// # Panics
        pub fn get_truncnorm_distribution(
            &self,
            mu: ArrayView1<NNVFloat>,
            sigma: ArrayView2<NNVFloat>,
            max_accept_reject_iters: usize,
            stability_eps: NNVFloat,
        ) -> GaussianDistribution {
            let mu = mu.mapv(std::convert::Into::into);
            let sigma = sigma.mapv(std::convert::Into::into);
            let mut constraint_coeffs: Array2<f64> = self.coeffs().mapv(std::convert::Into::into);
            let mut ub = self.rhs().mapv(std::convert::Into::into);
            let inv_coeffs = {
                let (u_opt, mut s, vt_opt) = constraint_coeffs.svd(true, true).unwrap();
                let s_max = *s.max().unwrap();
                s /= s_max;
                constraint_coeffs /= s_max;
                ub /= s_max;
                let s_matrix = {
                    let mut zeros = Array2::zeros([
                        vt_opt.as_ref().unwrap().shape()[0],
                        u_opt.as_ref().unwrap().shape()[1],
                    ]);
                    zeros.diag_mut().assign(&s);
                    zeros
                };
                vt_opt.unwrap().t().dot(&s_matrix).dot(&u_opt.unwrap().t())
            };
            let sq_constr_sigma = {
                let sigma: Array2<f64> = constraint_coeffs.dot(&sigma.dot(&constraint_coeffs.t()));
                let diag_addn: Array2<f64> =
                    Array2::from_diag(&Array1::from_elem(sigma.nrows(), stability_eps));
                sigma + diag_addn
            };
            let sq_ub = ub;
            let sq_constr_ub = &sq_ub - &constraint_coeffs.dot(&mu);
            let sq_constr_lb = Array1::from_elem(sq_constr_ub.len(), f64::NEG_INFINITY);
            let distribution = MultivariateTruncatedNormal::<Ix2>::new(
                mu,
                sq_constr_sigma,
                sq_constr_lb,
                sq_constr_ub,
                max_accept_reject_iters,
            );
            GaussianDistribution::TruncGaussian {
                distribution,
                inv_coeffs,
            }
        }
        /// # Panics
        /// Returns None if the reduced polytope is empty
        pub fn reduce_fixed_inputs(&self, bounds_opt: &Option<Vec<Bounds1>>) -> Option<Self> {
            if bounds_opt.is_none() {
                return Some(self.clone());
            }
            let bounds = bounds_opt
                .as_ref()
                .unwrap()
                .iter()
                .fold(Bounds1::default(0), Bounds1::append);
            let fixed_idxs = bounds.fixed_idxs();
            let fixed_vals = bounds.fixed_vals_or_zeros();
            self.reduce_with_values(fixed_vals.view(), fixed_idxs.view())
                .map(|mut reduced_poly| {
                    reduced_poly.filter_trivial();
                    reduced_poly
                })
        }
        /// Check whether the Star set is empty.
        ///
        /// This method assumes that the constraints bound each dimension,
        /// both lower and upper.
        ///
        /// # Panics
        pub fn is_empty(&self, bounds_opt: Option<&Bounds1>) -> bool {
            let c = Array1::ones(self.num_dims());
            let solved = solve(self.coeffs().rows(), self.rhs(), c.view(), bounds_opt);
            !match solved {
                LinearSolution::Solution(_, _) | LinearSolution::Unbounded(_) => true,
                _ => false,
            }
        }
    }
    /// Scale by scalar
    impl Mul<NNVFloat> for Polytope {
        type Output = Self;
        fn mul(self, rhs: NNVFloat) -> Self {
            Self {
                coeffs: self.coeffs * rhs,
                rhs: self.rhs * rhs,
            }
        }
    }
    /// Scale by scalar
    impl MulAssign<NNVFloat> for Polytope {
        fn mul_assign(&mut self, rhs: NNVFloat) {
            self.coeffs *= rhs;
            self.rhs *= rhs;
        }
    }
    impl From<Affine2> for Polytope {
        fn from(aff: Affine2) -> Self {
            Self {
                coeffs: aff.basis().to_owned(),
                rhs: aff.shift().to_owned(),
            }
        }
    }
}
pub mod star {
    #![allow(clippy::module_name_repetitions, clippy::similar_names, non_snake_case)]
    //! Implementation of [star sets](https://link.springer.com/chapter/10.1007/978-3-030-30942-8_39)
    //! for representing affine transformed sets
    use crate::affine::{Affine, Affine2, Affine4};
    use crate::bounds::Bounds1;
    use crate::gaussian::GaussianDistribution;
    use crate::lp::solve;
    use crate::lp::LinearSolution;
    use crate::polytope::Polytope;
    use crate::tensorshape::TensorShape;
    use crate::NNVFloat;
    use ndarray::array;
    use ndarray::ArrayView1;
    use ndarray::Dimension;
    use ndarray::Ix4;
    use ndarray::{Array1, Array2};
    use ndarray::{Array4, ArrayView2};
    use ndarray::{Axis, Ix2};
    use num::Float;
    use serde::{Deserialize, Serialize};
    use std::fmt::Debug;
    pub type Star2 = Star<Ix2>;
    pub type Star4 = Star<Ix4>;
    /// Representation of a set acted on by a deep neural network (DNN)
    ///
    /// Star sets are defined by a 1) constraint coefficient matrix, 2) upper
    /// bound vector, 3) basis matrix, 4) center vector. (1) and (2) define a
    /// polyhedron and (3) and (4) define an affine transformation of that
    /// polyhedron.
    ///
    /// Each Star set represents two sets implicitly: an input set and a
    /// representation set. The input set is defined in the input space of the deep
    /// neural network of interest. It's a polyhedron defined by the Star's
    /// constraints (coefficient matrix and upper bound vector). The representation
    /// set is defined in a latent or output space of the DNN. It is calculated by
    /// applying the affine transformation defined by the Star's basis and center
    /// to the input set polyhedron.
    ///
    /// Based on: Tran, Hoang-Dung, et al. "Star-based reachability analysis of
    /// deep neural networks." International Symposium on Formal Methods. Springer,
    /// Cham, 2019.
    pub struct Star<D: Dimension> {
        /// `representation` is the concatenation of [basis center] (where
        /// center is a column vector) and captures information about the
        /// transformed set
        representation: Affine<D>,
        /// `constraints` is the concatenation of [coeffs upper_bounds]
        /// and is a representation of the input polyhedron
        constraints: Option<Polytope>,
    }
    #[automatically_derived]
    #[allow(unused_qualifications)]
    impl<D: ::core::clone::Clone + Dimension> ::core::clone::Clone for Star<D> {
        #[inline]
        fn clone(&self) -> Star<D> {
            match *self {
                Star {
                    representation: ref __self_0_0,
                    constraints: ref __self_0_1,
                } => Star {
                    representation: ::core::clone::Clone::clone(&(*__self_0_0)),
                    constraints: ::core::clone::Clone::clone(&(*__self_0_1)),
                },
            }
        }
    }
    #[automatically_derived]
    #[allow(unused_qualifications)]
    impl<D: ::core::fmt::Debug + Dimension> ::core::fmt::Debug for Star<D> {
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            match *self {
                Star {
                    representation: ref __self_0_0,
                    constraints: ref __self_0_1,
                } => {
                    let debug_trait_builder = &mut ::core::fmt::Formatter::debug_struct(f, "Star");
                    let _ = ::core::fmt::DebugStruct::field(
                        debug_trait_builder,
                        "representation",
                        &&(*__self_0_0),
                    );
                    let _ = ::core::fmt::DebugStruct::field(
                        debug_trait_builder,
                        "constraints",
                        &&(*__self_0_1),
                    );
                    ::core::fmt::DebugStruct::finish(debug_trait_builder)
                }
            }
        }
    }
    #[doc(hidden)]
    #[allow(non_upper_case_globals, unused_attributes, unused_qualifications)]
    const _: () = {
        #[allow(unused_extern_crates, clippy::useless_attribute)]
        extern crate serde as _serde;
        #[automatically_derived]
        impl<'de, D: Dimension> _serde::Deserialize<'de> for Star<D>
        where
            D: _serde::Deserialize<'de>,
        {
            fn deserialize<__D>(__deserializer: __D) -> _serde::__private::Result<Self, __D::Error>
            where
                __D: _serde::Deserializer<'de>,
            {
                #[allow(non_camel_case_types)]
                enum __Field {
                    __field0,
                    __field1,
                    __ignore,
                }
                struct __FieldVisitor;
                impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                    type Value = __Field;
                    fn expecting(
                        &self,
                        __formatter: &mut _serde::__private::Formatter,
                    ) -> _serde::__private::fmt::Result {
                        _serde::__private::Formatter::write_str(__formatter, "field identifier")
                    }
                    fn visit_u64<__E>(
                        self,
                        __value: u64,
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            0u64 => _serde::__private::Ok(__Field::__field0),
                            1u64 => _serde::__private::Ok(__Field::__field1),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                    fn visit_str<__E>(
                        self,
                        __value: &str,
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            "representation" => _serde::__private::Ok(__Field::__field0),
                            "constraints" => _serde::__private::Ok(__Field::__field1),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                    fn visit_bytes<__E>(
                        self,
                        __value: &[u8],
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            b"representation" => _serde::__private::Ok(__Field::__field0),
                            b"constraints" => _serde::__private::Ok(__Field::__field1),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                }
                impl<'de> _serde::Deserialize<'de> for __Field {
                    #[inline]
                    fn deserialize<__D>(
                        __deserializer: __D,
                    ) -> _serde::__private::Result<Self, __D::Error>
                    where
                        __D: _serde::Deserializer<'de>,
                    {
                        _serde::Deserializer::deserialize_identifier(__deserializer, __FieldVisitor)
                    }
                }
                struct __Visitor<'de, D: Dimension>
                where
                    D: _serde::Deserialize<'de>,
                {
                    marker: _serde::__private::PhantomData<Star<D>>,
                    lifetime: _serde::__private::PhantomData<&'de ()>,
                }
                impl<'de, D: Dimension> _serde::de::Visitor<'de> for __Visitor<'de, D>
                where
                    D: _serde::Deserialize<'de>,
                {
                    type Value = Star<D>;
                    fn expecting(
                        &self,
                        __formatter: &mut _serde::__private::Formatter,
                    ) -> _serde::__private::fmt::Result {
                        _serde::__private::Formatter::write_str(__formatter, "struct Star")
                    }
                    #[inline]
                    fn visit_seq<__A>(
                        self,
                        mut __seq: __A,
                    ) -> _serde::__private::Result<Self::Value, __A::Error>
                    where
                        __A: _serde::de::SeqAccess<'de>,
                    {
                        let __field0 = match match _serde::de::SeqAccess::next_element::<Affine<D>>(
                            &mut __seq,
                        ) {
                            _serde::__private::Ok(__val) => __val,
                            _serde::__private::Err(__err) => {
                                return _serde::__private::Err(__err);
                            }
                        } {
                            _serde::__private::Some(__value) => __value,
                            _serde::__private::None => {
                                return _serde::__private::Err(_serde::de::Error::invalid_length(
                                    0usize,
                                    &"struct Star with 2 elements",
                                ));
                            }
                        };
                        let __field1 = match match _serde::de::SeqAccess::next_element::<
                            Option<Polytope>,
                        >(&mut __seq)
                        {
                            _serde::__private::Ok(__val) => __val,
                            _serde::__private::Err(__err) => {
                                return _serde::__private::Err(__err);
                            }
                        } {
                            _serde::__private::Some(__value) => __value,
                            _serde::__private::None => {
                                return _serde::__private::Err(_serde::de::Error::invalid_length(
                                    1usize,
                                    &"struct Star with 2 elements",
                                ));
                            }
                        };
                        _serde::__private::Ok(Star {
                            representation: __field0,
                            constraints: __field1,
                        })
                    }
                    #[inline]
                    fn visit_map<__A>(
                        self,
                        mut __map: __A,
                    ) -> _serde::__private::Result<Self::Value, __A::Error>
                    where
                        __A: _serde::de::MapAccess<'de>,
                    {
                        let mut __field0: _serde::__private::Option<Affine<D>> =
                            _serde::__private::None;
                        let mut __field1: _serde::__private::Option<Option<Polytope>> =
                            _serde::__private::None;
                        while let _serde::__private::Some(__key) =
                            match _serde::de::MapAccess::next_key::<__Field>(&mut __map) {
                                _serde::__private::Ok(__val) => __val,
                                _serde::__private::Err(__err) => {
                                    return _serde::__private::Err(__err);
                                }
                            }
                        {
                            match __key {
                                __Field::__field0 => {
                                    if _serde::__private::Option::is_some(&__field0) {
                                        return _serde::__private::Err(
                                            <__A::Error as _serde::de::Error>::duplicate_field(
                                                "representation",
                                            ),
                                        );
                                    }
                                    __field0 = _serde::__private::Some(
                                        match _serde::de::MapAccess::next_value::<Affine<D>>(
                                            &mut __map,
                                        ) {
                                            _serde::__private::Ok(__val) => __val,
                                            _serde::__private::Err(__err) => {
                                                return _serde::__private::Err(__err);
                                            }
                                        },
                                    );
                                }
                                __Field::__field1 => {
                                    if _serde::__private::Option::is_some(&__field1) {
                                        return _serde::__private::Err(
                                            <__A::Error as _serde::de::Error>::duplicate_field(
                                                "constraints",
                                            ),
                                        );
                                    }
                                    __field1 = _serde::__private::Some(
                                        match _serde::de::MapAccess::next_value::<Option<Polytope>>(
                                            &mut __map,
                                        ) {
                                            _serde::__private::Ok(__val) => __val,
                                            _serde::__private::Err(__err) => {
                                                return _serde::__private::Err(__err);
                                            }
                                        },
                                    );
                                }
                                _ => {
                                    let _ = match _serde::de::MapAccess::next_value::<
                                        _serde::de::IgnoredAny,
                                    >(&mut __map)
                                    {
                                        _serde::__private::Ok(__val) => __val,
                                        _serde::__private::Err(__err) => {
                                            return _serde::__private::Err(__err);
                                        }
                                    };
                                }
                            }
                        }
                        let __field0 = match __field0 {
                            _serde::__private::Some(__field0) => __field0,
                            _serde::__private::None => {
                                match _serde::__private::de::missing_field("representation") {
                                    _serde::__private::Ok(__val) => __val,
                                    _serde::__private::Err(__err) => {
                                        return _serde::__private::Err(__err);
                                    }
                                }
                            }
                        };
                        let __field1 = match __field1 {
                            _serde::__private::Some(__field1) => __field1,
                            _serde::__private::None => {
                                match _serde::__private::de::missing_field("constraints") {
                                    _serde::__private::Ok(__val) => __val,
                                    _serde::__private::Err(__err) => {
                                        return _serde::__private::Err(__err);
                                    }
                                }
                            }
                        };
                        _serde::__private::Ok(Star {
                            representation: __field0,
                            constraints: __field1,
                        })
                    }
                }
                const FIELDS: &'static [&'static str] = &["representation", "constraints"];
                _serde::Deserializer::deserialize_struct(
                    __deserializer,
                    "Star",
                    FIELDS,
                    __Visitor {
                        marker: _serde::__private::PhantomData::<Star<D>>,
                        lifetime: _serde::__private::PhantomData,
                    },
                )
            }
        }
    };
    #[doc(hidden)]
    #[allow(non_upper_case_globals, unused_attributes, unused_qualifications)]
    const _: () = {
        #[allow(unused_extern_crates, clippy::useless_attribute)]
        extern crate serde as _serde;
        #[automatically_derived]
        impl<D: Dimension> _serde::Serialize for Star<D>
        where
            D: _serde::Serialize,
        {
            fn serialize<__S>(
                &self,
                __serializer: __S,
            ) -> _serde::__private::Result<__S::Ok, __S::Error>
            where
                __S: _serde::Serializer,
            {
                let mut __serde_state = match _serde::Serializer::serialize_struct(
                    __serializer,
                    "Star",
                    false as usize + 1 + 1,
                ) {
                    _serde::__private::Ok(__val) => __val,
                    _serde::__private::Err(__err) => {
                        return _serde::__private::Err(__err);
                    }
                };
                match _serde::ser::SerializeStruct::serialize_field(
                    &mut __serde_state,
                    "representation",
                    &self.representation,
                ) {
                    _serde::__private::Ok(__val) => __val,
                    _serde::__private::Err(__err) => {
                        return _serde::__private::Err(__err);
                    }
                };
                match _serde::ser::SerializeStruct::serialize_field(
                    &mut __serde_state,
                    "constraints",
                    &self.constraints,
                ) {
                    _serde::__private::Ok(__val) => __val,
                    _serde::__private::Err(__err) => {
                        return _serde::__private::Err(__err);
                    }
                };
                _serde::ser::SerializeStruct::end(__serde_state)
            }
        }
    };
    impl<D: Dimension> Star<D> {
        pub fn ndim(&self) -> usize {
            self.representation.ndim()
        }
        pub fn input_space_polytope(&self) -> Option<&Polytope> {
            self.constraints.as_ref()
        }
        pub fn center(&self) -> ArrayView1<NNVFloat> {
            self.representation.shift()
        }
        pub fn get_representation(&self) -> &Affine<D> {
            &self.representation
        }
    }
    impl<D: Dimension> Star<D> {
        pub fn num_constraints(&self) -> usize {
            match &self.constraints {
                Some(polytope) => polytope.num_constraints(),
                None => 0,
            }
        }
        /// Add constraints to restrict the input set. Each row represents a
        /// constraint and the last column represents the upper bounds.
        #[must_use]
        pub fn add_constraint(mut self, coeffs: ArrayView1<NNVFloat>, rhs: NNVFloat) -> Self {
            if let Some(ref mut constrs) = self.constraints {
                constrs.add_eqn(coeffs, rhs);
            } else {
                self.constraints =
                    Polytope::nonempty_new(&coeffs.to_owned().insert_axis(Axis(0)), &{
                        ::ndarray::Array::from(<[_]>::into_vec(box [rhs]))
                    });
            }
            self
        }
        #[must_use]
        /// # Panics
        pub fn remove_constraint(mut self, idx: usize) -> Self {
            self.constraints.as_mut().map_or_else(
                || {
                    ::core::panicking::panic("explicit panic");
                },
                |constrs| constrs.remove_eqn(idx),
            );
            self
        }
    }
    impl Star2 {
        pub fn get_constraint_coeffs(&self) -> Option<Array2<NNVFloat>> {
            self.constraints.as_ref().map(|x| x.coeffs().to_owned())
        }
        #[must_use]
        pub fn get_safe_subset(&self, safe_value: NNVFloat) -> Self {
            let mut subset = self.clone();
            let mut new_constr: Polytope = self.representation.clone().into();
            let mut rhs = new_constr.rhs_mut();
            rhs *= -1.;
            rhs += safe_value;
            subset.intersect_input(&new_constr);
            subset
        }
        pub fn intersect_input(&mut self, other: &Polytope) {
            self.constraints = self
                .constraints
                .as_ref()
                .map_or(Some(other.clone()), |x| Some(x.intersect(other)));
        }
        /// # Panics
        /// TODO
        pub fn get_input_trunc_gaussian(
            &self,
            mu: ArrayView1<NNVFloat>,
            sigma: ArrayView2<NNVFloat>,
            max_accept_reject_iters: usize,
            stability_eps: NNVFloat,
            bounds_opt: &Option<Bounds1>,
        ) -> Option<GaussianDistribution> {
            ::core::panicking::panic("not yet implemented");
        }
        /// Create a new Star with given dimension.
        ///
        /// By default this Star covers the space because it has no constraints. To add constraints call `.add_constraints`.
        ///
        /// # Panics
        pub fn default(input_shape: &TensorShape) -> Self {
            if true {
                match (&input_shape.rank(), &1) {
                    (left_val, right_val) => {
                        if !(*left_val == *right_val) {
                            let kind = ::core::panicking::AssertKind::Eq;
                            ::core::panicking::assert_failed(
                                kind,
                                &*left_val,
                                &*right_val,
                                ::core::option::Option::None,
                            );
                        }
                    }
                };
            };
            if true {
                if !input_shape.is_fully_defined() {
                    ::core::panicking::panic("assertion failed: input_shape.is_fully_defined()")
                };
            };
            let dim = input_shape[0].unwrap();
            Self {
                representation: Affine2::new(Array2::eye(dim), Array1::zeros(dim)),
                constraints: None,
            }
        }
        /// Create a new Star with given basis vector and center.
        ///
        /// By default this Star covers the space because it has no constraints. To add constraints call `.add_constraints`.
        pub fn new(basis: Array2<NNVFloat>, center: Array1<NNVFloat>) -> Self {
            Self {
                representation: Affine2::new(basis, center),
                constraints: None,
            }
        }
        /// # Panics
        #[must_use]
        pub fn with_constraints(mut self, constraints: Polytope) -> Self {
            if true {
                if !self.constraints.is_none() {
                    ::core::panicking::panic_fmt(::core::fmt::Arguments::new_v1(
                        &["explicit panic"],
                        &[],
                    ))
                };
            };
            self.constraints = Some(constraints);
            self
        }
        /// Get the dimension of the input space
        pub fn input_dim(&self) -> usize {
            self.representation.input_dim()
        }
        /// Get the dimension of the representation space
        pub fn representation_space_dim(&self) -> usize {
            self.representation.output_dim()
        }
    }
    impl Star2 {
        /// Apply an affine transformation to the representation
        #[must_use]
        pub fn affine_map2(&self, affine: &Affine<Ix2>) -> Self {
            Self {
                representation: affine * &self.representation,
                constraints: self.constraints.clone(),
            }
        }
        pub fn step_relu2(
            &self,
            index: usize,
            input_bounds_opt: Option<&Bounds1>,
        ) -> (Option<Self>, Option<Self>) {
            let (coeffs, shift) = {
                let aff = self.representation.get_eqn(index);
                let neg_basis_part: Array2<NNVFloat> = &aff.basis() * -1.;
                let shift = aff.shift();
                (neg_basis_part.row(0).to_owned(), shift[[0]])
            };
            let upper_star = self.clone().add_constraint(coeffs.view(), shift);
            let mut lower_star = self.clone().add_constraint((&coeffs * -1.).view(), -shift);
            lower_star.representation.zero_eqn(index);
            let lower_star_opt = if lower_star.is_empty(input_bounds_opt) {
                None
            } else {
                Some(lower_star)
            };
            let upper_star_opt = if upper_star.is_empty(input_bounds_opt) {
                None
            } else {
                Some(upper_star)
            };
            (lower_star_opt, upper_star_opt)
        }
        pub fn step_relu2_dropout(
            &self,
            index: usize,
            input_bounds_opt: Option<&Bounds1>,
        ) -> (Option<Self>, Option<Self>, Option<Self>) {
            let mut dropout_star = self.clone();
            dropout_star.representation.zero_eqn(index);
            let stars = self.step_relu2(index, input_bounds_opt);
            let dropout_star_opt = if dropout_star.is_empty(input_bounds_opt) {
                None
            } else {
                Some(dropout_star)
            };
            (dropout_star_opt, stars.0, stars.1)
        }
        /// Calculates the minimum value of the equation at index `idx`
        /// given the constraints
        ///
        /// This method assumes that the constraints bound each dimension,
        /// both lower and upper.
        ///
        /// # Panics
        /// TODO: Change output type to Option<T>
        ///
        /// TODO: `ResolutionError::Unbounded` can result whether or not the
        /// constraints are infeasible if there are zeros in the
        /// objective. This needs to be checked either here or in the
        /// solve function. Currently this is way too hard to do, so we
        /// panic instead. We have an assumption that we start with a
        /// bounded box and therefore should never be unbounded.
        pub fn get_output_min(&self, idx: usize, input_bounds: &Bounds1) -> NNVFloat {
            let eqn = self.representation.get_eqn(idx);
            let shift = eqn.shift()[0];
            self.constraints.as_ref().map_or_else(
                || {
                    crate::util::signed_dot(
                        &eqn.basis(),
                        &input_bounds.lower(),
                        &input_bounds.upper(),
                    )[[0]]
                        + shift
                },
                |poly| {
                    let solved = solve(
                        poly.coeffs().rows(),
                        poly.rhs(),
                        eqn.basis().index_axis(Axis(0), 0),
                        Some(input_bounds),
                    );
                    if let LinearSolution::Solution(_, val) = solved {
                        shift + val
                    } else if let LinearSolution::Unbounded(_) = solved {
                        NNVFloat::neg_infinity()
                    } else {
                        ::core::panicking::panic_fmt(::core::fmt::Arguments::new_v1(
                            &["Solution: "],
                            &[::core::fmt::ArgumentV1::new_debug(&solved)],
                        ))
                    }
                },
            )
        }
        /// Calculates the maximum value of the equation at index `idx`
        /// given the constraints
        ///
        /// This method assumes that the constraints bound each dimension,
        /// both lower and upper.
        ///
        /// # Panics
        /// TODO: Change output type to Option<T>
        pub fn get_output_max(&self, idx: usize, input_bounds: &Bounds1) -> NNVFloat {
            let eqn = self.representation.get_eqn(idx);
            let shift = eqn.shift()[0];
            self.constraints.as_ref().map_or_else(
                || {
                    crate::util::signed_dot(
                        &eqn.basis(),
                        &input_bounds.upper(),
                        &input_bounds.lower(),
                    )[[0]]
                        + shift
                },
                |poly| {
                    let solved = solve(
                        poly.coeffs().rows(),
                        poly.rhs(),
                        eqn.basis().index_axis(Axis(0), 0).mapv(|x| x * -1.).view(),
                        Some(input_bounds),
                    );
                    if let LinearSolution::Solution(_, val) = solved {
                        shift - val
                    } else if let LinearSolution::Unbounded(_) = solved {
                        NNVFloat::infinity()
                    } else {
                        ::core::panicking::panic("explicit panic")
                    }
                },
            )
        }
        /// # Panics
        pub fn can_maximize_output_idx(&self, class_idx: usize) -> bool {
            let class_eqn = self.representation.get_eqn(class_idx);
            let (class_coeffs, class_shift): (ArrayView1<NNVFloat>, NNVFloat) = {
                (
                    class_eqn.basis().remove_axis(Axis(0)),
                    class_eqn.shift()[[0]],
                )
            };
            let nvars = class_coeffs.len();
            if self.constraints.is_none() {
                return true;
            }
            let poly = self.constraints.as_ref().unwrap();
            let block_coeffs = poly.coeffs();
            let (A, b) = (block_coeffs.rows(), poly.rhs());
            let mut coeffs = Vec::new();
            let mut shifts = Vec::new();
            for idx in 0..self.representation.shift().ndim() {
                if idx == class_idx {
                    continue;
                }
                {
                    let (diff_coeffs, diff_shift) = {
                        let other_class_eqn = self.representation.get_eqn(idx);
                        (
                            &other_class_eqn.basis().row(0) - &class_coeffs,
                            class_shift - other_class_eqn.shift()[[0]],
                        )
                    };
                    coeffs.push(diff_coeffs);
                    shifts.push(diff_shift);
                }
            }
            let solve_a = A
                .into_iter()
                .chain(coeffs.iter().map(ndarray::ArrayBase::view));
            let solved = solve(
                solve_a,
                b.iter().chain(shifts.iter()),
                Array1::ones(nvars).view(),
                Some(&Bounds1::trivial(nvars)),
            );
            match solved {
                LinearSolution::Solution(..) | LinearSolution::Unbounded(..) => true,
                _ => false,
            }
        }
        pub fn calculate_output_axis_aligned_bounding_box(
            &self,
            input_outer_bounds: &Bounds1,
        ) -> Bounds1 {
            let lbs = Array1::from_iter(
                (0..self.representation_space_dim())
                    .map(|x| self.get_output_min(x, input_outer_bounds)),
            );
            let ubs = Array1::from_iter(
                (0..self.representation_space_dim())
                    .map(|x| self.get_output_max(x, input_outer_bounds)),
            );
            Bounds1::new(lbs.view(), ubs.view())
        }
        /// Check whether the Star set is empty.
        pub fn is_empty(&self, input_bounds_opt: Option<&Bounds1>) -> bool {
            self.constraints
                .as_ref()
                .map_or(false, |x| x.is_empty(input_bounds_opt))
        }
    }
    impl Star4 {
        /// Create a new Star with given dimension.
        ///
        /// By default this Star covers the space because it has no constraints. To add constraints call `.add_constraints`.
        ///
        /// # Panics
        pub fn default(input_shape: &TensorShape) -> Self {
            if true {
                match (&input_shape.rank(), &3) {
                    (left_val, right_val) => {
                        if !(*left_val == *right_val) {
                            let kind = ::core::panicking::AssertKind::Eq;
                            ::core::panicking::assert_failed(
                                kind,
                                &*left_val,
                                &*right_val,
                                ::core::option::Option::None,
                            );
                        }
                    }
                };
            };
            if true {
                if !input_shape.is_fully_defined() {
                    ::core::panicking::panic("assertion failed: input_shape.is_fully_defined()")
                };
            };
            let shape_slice = input_shape.as_defined_slice().unwrap();
            let slice_exact: [usize; 4] = [
                shape_slice[0],
                shape_slice[1],
                shape_slice[2],
                shape_slice[3],
            ];
            Self {
                representation: Affine4::new(
                    Array4::ones(slice_exact),
                    Array1::zeros(shape_slice[3]),
                ),
                constraints: None,
            }
        }
    }
}
pub mod star_node {
    #![allow(clippy::module_name_repetitions)]
    use crate::bounds::Bounds1;
    use crate::gaussian::GaussianDistribution;
    use crate::graph::OperationId;
    use crate::num::Float;
    use crate::polytope::Polytope;
    use crate::star::Star;
    use crate::NNVFloat;
    use ndarray::Array1;
    use ndarray::ArrayView1;
    use ndarray::ArrayView2;
    use ndarray::Dimension;
    use ndarray::Ix2;
    use rand::Rng;
    use serde::{Deserialize, Serialize};
    use std::fmt::Debug;
    use truncnorm::tilting::TiltingSolution;
    /// # Assumptions:
    /// children: Option<Vec<StarNodeId>>: None if not expanded.
    ///                           Empty if actually no children, terminal node (does not necessarily mean node is an output).
    ///                           1 node for many different options (affine, single child steprelu, etc.)
    ///                           Multiple children if adding partition constraints.
    pub struct StarNodeRelationship {
        pub operation_id: OperationId,
        pub step: Option<usize>,
        pub input_node_ids: Vec<usize>,
        pub output_node_ids: Option<Vec<usize>>,
    }
    #[automatically_derived]
    #[allow(unused_qualifications)]
    impl ::core::fmt::Debug for StarNodeRelationship {
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            match *self {
                StarNodeRelationship {
                    operation_id: ref __self_0_0,
                    step: ref __self_0_1,
                    input_node_ids: ref __self_0_2,
                    output_node_ids: ref __self_0_3,
                } => {
                    let debug_trait_builder =
                        &mut ::core::fmt::Formatter::debug_struct(f, "StarNodeRelationship");
                    let _ = ::core::fmt::DebugStruct::field(
                        debug_trait_builder,
                        "operation_id",
                        &&(*__self_0_0),
                    );
                    let _ = ::core::fmt::DebugStruct::field(
                        debug_trait_builder,
                        "step",
                        &&(*__self_0_1),
                    );
                    let _ = ::core::fmt::DebugStruct::field(
                        debug_trait_builder,
                        "input_node_ids",
                        &&(*__self_0_2),
                    );
                    let _ = ::core::fmt::DebugStruct::field(
                        debug_trait_builder,
                        "output_node_ids",
                        &&(*__self_0_3),
                    );
                    ::core::fmt::DebugStruct::finish(debug_trait_builder)
                }
            }
        }
    }
    #[automatically_derived]
    #[allow(unused_qualifications)]
    impl ::core::clone::Clone for StarNodeRelationship {
        #[inline]
        fn clone(&self) -> StarNodeRelationship {
            match *self {
                StarNodeRelationship {
                    operation_id: ref __self_0_0,
                    step: ref __self_0_1,
                    input_node_ids: ref __self_0_2,
                    output_node_ids: ref __self_0_3,
                } => StarNodeRelationship {
                    operation_id: ::core::clone::Clone::clone(&(*__self_0_0)),
                    step: ::core::clone::Clone::clone(&(*__self_0_1)),
                    input_node_ids: ::core::clone::Clone::clone(&(*__self_0_2)),
                    output_node_ids: ::core::clone::Clone::clone(&(*__self_0_3)),
                },
            }
        }
    }
    #[automatically_derived]
    #[allow(unused_qualifications)]
    impl ::core::default::Default for StarNodeRelationship {
        #[inline]
        fn default() -> StarNodeRelationship {
            StarNodeRelationship {
                operation_id: ::core::default::Default::default(),
                step: ::core::default::Default::default(),
                input_node_ids: ::core::default::Default::default(),
                output_node_ids: ::core::default::Default::default(),
            }
        }
    }
    #[doc(hidden)]
    #[allow(non_upper_case_globals, unused_attributes, unused_qualifications)]
    const _: () = {
        #[allow(unused_extern_crates, clippy::useless_attribute)]
        extern crate serde as _serde;
        #[automatically_derived]
        impl _serde::Serialize for StarNodeRelationship {
            fn serialize<__S>(
                &self,
                __serializer: __S,
            ) -> _serde::__private::Result<__S::Ok, __S::Error>
            where
                __S: _serde::Serializer,
            {
                let mut __serde_state = match _serde::Serializer::serialize_struct(
                    __serializer,
                    "StarNodeRelationship",
                    false as usize + 1 + 1 + 1 + 1,
                ) {
                    _serde::__private::Ok(__val) => __val,
                    _serde::__private::Err(__err) => {
                        return _serde::__private::Err(__err);
                    }
                };
                match _serde::ser::SerializeStruct::serialize_field(
                    &mut __serde_state,
                    "operation_id",
                    &self.operation_id,
                ) {
                    _serde::__private::Ok(__val) => __val,
                    _serde::__private::Err(__err) => {
                        return _serde::__private::Err(__err);
                    }
                };
                match _serde::ser::SerializeStruct::serialize_field(
                    &mut __serde_state,
                    "step",
                    &self.step,
                ) {
                    _serde::__private::Ok(__val) => __val,
                    _serde::__private::Err(__err) => {
                        return _serde::__private::Err(__err);
                    }
                };
                match _serde::ser::SerializeStruct::serialize_field(
                    &mut __serde_state,
                    "input_node_ids",
                    &self.input_node_ids,
                ) {
                    _serde::__private::Ok(__val) => __val,
                    _serde::__private::Err(__err) => {
                        return _serde::__private::Err(__err);
                    }
                };
                match _serde::ser::SerializeStruct::serialize_field(
                    &mut __serde_state,
                    "output_node_ids",
                    &self.output_node_ids,
                ) {
                    _serde::__private::Ok(__val) => __val,
                    _serde::__private::Err(__err) => {
                        return _serde::__private::Err(__err);
                    }
                };
                _serde::ser::SerializeStruct::end(__serde_state)
            }
        }
    };
    #[doc(hidden)]
    #[allow(non_upper_case_globals, unused_attributes, unused_qualifications)]
    const _: () = {
        #[allow(unused_extern_crates, clippy::useless_attribute)]
        extern crate serde as _serde;
        #[automatically_derived]
        impl<'de> _serde::Deserialize<'de> for StarNodeRelationship {
            fn deserialize<__D>(__deserializer: __D) -> _serde::__private::Result<Self, __D::Error>
            where
                __D: _serde::Deserializer<'de>,
            {
                #[allow(non_camel_case_types)]
                enum __Field {
                    __field0,
                    __field1,
                    __field2,
                    __field3,
                    __ignore,
                }
                struct __FieldVisitor;
                impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                    type Value = __Field;
                    fn expecting(
                        &self,
                        __formatter: &mut _serde::__private::Formatter,
                    ) -> _serde::__private::fmt::Result {
                        _serde::__private::Formatter::write_str(__formatter, "field identifier")
                    }
                    fn visit_u64<__E>(
                        self,
                        __value: u64,
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            0u64 => _serde::__private::Ok(__Field::__field0),
                            1u64 => _serde::__private::Ok(__Field::__field1),
                            2u64 => _serde::__private::Ok(__Field::__field2),
                            3u64 => _serde::__private::Ok(__Field::__field3),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                    fn visit_str<__E>(
                        self,
                        __value: &str,
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            "operation_id" => _serde::__private::Ok(__Field::__field0),
                            "step" => _serde::__private::Ok(__Field::__field1),
                            "input_node_ids" => _serde::__private::Ok(__Field::__field2),
                            "output_node_ids" => _serde::__private::Ok(__Field::__field3),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                    fn visit_bytes<__E>(
                        self,
                        __value: &[u8],
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            b"operation_id" => _serde::__private::Ok(__Field::__field0),
                            b"step" => _serde::__private::Ok(__Field::__field1),
                            b"input_node_ids" => _serde::__private::Ok(__Field::__field2),
                            b"output_node_ids" => _serde::__private::Ok(__Field::__field3),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                }
                impl<'de> _serde::Deserialize<'de> for __Field {
                    #[inline]
                    fn deserialize<__D>(
                        __deserializer: __D,
                    ) -> _serde::__private::Result<Self, __D::Error>
                    where
                        __D: _serde::Deserializer<'de>,
                    {
                        _serde::Deserializer::deserialize_identifier(__deserializer, __FieldVisitor)
                    }
                }
                struct __Visitor<'de> {
                    marker: _serde::__private::PhantomData<StarNodeRelationship>,
                    lifetime: _serde::__private::PhantomData<&'de ()>,
                }
                impl<'de> _serde::de::Visitor<'de> for __Visitor<'de> {
                    type Value = StarNodeRelationship;
                    fn expecting(
                        &self,
                        __formatter: &mut _serde::__private::Formatter,
                    ) -> _serde::__private::fmt::Result {
                        _serde::__private::Formatter::write_str(
                            __formatter,
                            "struct StarNodeRelationship",
                        )
                    }
                    #[inline]
                    fn visit_seq<__A>(
                        self,
                        mut __seq: __A,
                    ) -> _serde::__private::Result<Self::Value, __A::Error>
                    where
                        __A: _serde::de::SeqAccess<'de>,
                    {
                        let __field0 = match match _serde::de::SeqAccess::next_element::<OperationId>(
                            &mut __seq,
                        ) {
                            _serde::__private::Ok(__val) => __val,
                            _serde::__private::Err(__err) => {
                                return _serde::__private::Err(__err);
                            }
                        } {
                            _serde::__private::Some(__value) => __value,
                            _serde::__private::None => {
                                return _serde::__private::Err(_serde::de::Error::invalid_length(
                                    0usize,
                                    &"struct StarNodeRelationship with 4 elements",
                                ));
                            }
                        };
                        let __field1 = match match _serde::de::SeqAccess::next_element::<
                            Option<usize>,
                        >(&mut __seq)
                        {
                            _serde::__private::Ok(__val) => __val,
                            _serde::__private::Err(__err) => {
                                return _serde::__private::Err(__err);
                            }
                        } {
                            _serde::__private::Some(__value) => __value,
                            _serde::__private::None => {
                                return _serde::__private::Err(_serde::de::Error::invalid_length(
                                    1usize,
                                    &"struct StarNodeRelationship with 4 elements",
                                ));
                            }
                        };
                        let __field2 = match match _serde::de::SeqAccess::next_element::<Vec<usize>>(
                            &mut __seq,
                        ) {
                            _serde::__private::Ok(__val) => __val,
                            _serde::__private::Err(__err) => {
                                return _serde::__private::Err(__err);
                            }
                        } {
                            _serde::__private::Some(__value) => __value,
                            _serde::__private::None => {
                                return _serde::__private::Err(_serde::de::Error::invalid_length(
                                    2usize,
                                    &"struct StarNodeRelationship with 4 elements",
                                ));
                            }
                        };
                        let __field3 = match match _serde::de::SeqAccess::next_element::<
                            Option<Vec<usize>>,
                        >(&mut __seq)
                        {
                            _serde::__private::Ok(__val) => __val,
                            _serde::__private::Err(__err) => {
                                return _serde::__private::Err(__err);
                            }
                        } {
                            _serde::__private::Some(__value) => __value,
                            _serde::__private::None => {
                                return _serde::__private::Err(_serde::de::Error::invalid_length(
                                    3usize,
                                    &"struct StarNodeRelationship with 4 elements",
                                ));
                            }
                        };
                        _serde::__private::Ok(StarNodeRelationship {
                            operation_id: __field0,
                            step: __field1,
                            input_node_ids: __field2,
                            output_node_ids: __field3,
                        })
                    }
                    #[inline]
                    fn visit_map<__A>(
                        self,
                        mut __map: __A,
                    ) -> _serde::__private::Result<Self::Value, __A::Error>
                    where
                        __A: _serde::de::MapAccess<'de>,
                    {
                        let mut __field0: _serde::__private::Option<OperationId> =
                            _serde::__private::None;
                        let mut __field1: _serde::__private::Option<Option<usize>> =
                            _serde::__private::None;
                        let mut __field2: _serde::__private::Option<Vec<usize>> =
                            _serde::__private::None;
                        let mut __field3: _serde::__private::Option<Option<Vec<usize>>> =
                            _serde::__private::None;
                        while let _serde::__private::Some(__key) =
                            match _serde::de::MapAccess::next_key::<__Field>(&mut __map) {
                                _serde::__private::Ok(__val) => __val,
                                _serde::__private::Err(__err) => {
                                    return _serde::__private::Err(__err);
                                }
                            }
                        {
                            match __key {
                                __Field::__field0 => {
                                    if _serde::__private::Option::is_some(&__field0) {
                                        return _serde::__private::Err(
                                            <__A::Error as _serde::de::Error>::duplicate_field(
                                                "operation_id",
                                            ),
                                        );
                                    }
                                    __field0 = _serde::__private::Some(
                                        match _serde::de::MapAccess::next_value::<OperationId>(
                                            &mut __map,
                                        ) {
                                            _serde::__private::Ok(__val) => __val,
                                            _serde::__private::Err(__err) => {
                                                return _serde::__private::Err(__err);
                                            }
                                        },
                                    );
                                }
                                __Field::__field1 => {
                                    if _serde::__private::Option::is_some(&__field1) {
                                        return _serde::__private::Err(
                                            <__A::Error as _serde::de::Error>::duplicate_field(
                                                "step",
                                            ),
                                        );
                                    }
                                    __field1 = _serde::__private::Some(
                                        match _serde::de::MapAccess::next_value::<Option<usize>>(
                                            &mut __map,
                                        ) {
                                            _serde::__private::Ok(__val) => __val,
                                            _serde::__private::Err(__err) => {
                                                return _serde::__private::Err(__err);
                                            }
                                        },
                                    );
                                }
                                __Field::__field2 => {
                                    if _serde::__private::Option::is_some(&__field2) {
                                        return _serde::__private::Err(
                                            <__A::Error as _serde::de::Error>::duplicate_field(
                                                "input_node_ids",
                                            ),
                                        );
                                    }
                                    __field2 = _serde::__private::Some(
                                        match _serde::de::MapAccess::next_value::<Vec<usize>>(
                                            &mut __map,
                                        ) {
                                            _serde::__private::Ok(__val) => __val,
                                            _serde::__private::Err(__err) => {
                                                return _serde::__private::Err(__err);
                                            }
                                        },
                                    );
                                }
                                __Field::__field3 => {
                                    if _serde::__private::Option::is_some(&__field3) {
                                        return _serde::__private::Err(
                                            <__A::Error as _serde::de::Error>::duplicate_field(
                                                "output_node_ids",
                                            ),
                                        );
                                    }
                                    __field3 = _serde::__private::Some(
                                        match _serde::de::MapAccess::next_value::<Option<Vec<usize>>>(
                                            &mut __map,
                                        ) {
                                            _serde::__private::Ok(__val) => __val,
                                            _serde::__private::Err(__err) => {
                                                return _serde::__private::Err(__err);
                                            }
                                        },
                                    );
                                }
                                _ => {
                                    let _ = match _serde::de::MapAccess::next_value::<
                                        _serde::de::IgnoredAny,
                                    >(&mut __map)
                                    {
                                        _serde::__private::Ok(__val) => __val,
                                        _serde::__private::Err(__err) => {
                                            return _serde::__private::Err(__err);
                                        }
                                    };
                                }
                            }
                        }
                        let __field0 = match __field0 {
                            _serde::__private::Some(__field0) => __field0,
                            _serde::__private::None => {
                                match _serde::__private::de::missing_field("operation_id") {
                                    _serde::__private::Ok(__val) => __val,
                                    _serde::__private::Err(__err) => {
                                        return _serde::__private::Err(__err);
                                    }
                                }
                            }
                        };
                        let __field1 = match __field1 {
                            _serde::__private::Some(__field1) => __field1,
                            _serde::__private::None => {
                                match _serde::__private::de::missing_field("step") {
                                    _serde::__private::Ok(__val) => __val,
                                    _serde::__private::Err(__err) => {
                                        return _serde::__private::Err(__err);
                                    }
                                }
                            }
                        };
                        let __field2 = match __field2 {
                            _serde::__private::Some(__field2) => __field2,
                            _serde::__private::None => {
                                match _serde::__private::de::missing_field("input_node_ids") {
                                    _serde::__private::Ok(__val) => __val,
                                    _serde::__private::Err(__err) => {
                                        return _serde::__private::Err(__err);
                                    }
                                }
                            }
                        };
                        let __field3 = match __field3 {
                            _serde::__private::Some(__field3) => __field3,
                            _serde::__private::None => {
                                match _serde::__private::de::missing_field("output_node_ids") {
                                    _serde::__private::Ok(__val) => __val,
                                    _serde::__private::Err(__err) => {
                                        return _serde::__private::Err(__err);
                                    }
                                }
                            }
                        };
                        _serde::__private::Ok(StarNodeRelationship {
                            operation_id: __field0,
                            step: __field1,
                            input_node_ids: __field2,
                            output_node_ids: __field3,
                        })
                    }
                }
                const FIELDS: &'static [&'static str] =
                    &["operation_id", "step", "input_node_ids", "output_node_ids"];
                _serde::Deserializer::deserialize_struct(
                    __deserializer,
                    "StarNodeRelationship",
                    FIELDS,
                    __Visitor {
                        marker: _serde::__private::PhantomData::<StarNodeRelationship>,
                        lifetime: _serde::__private::PhantomData,
                    },
                )
            }
        }
    };
    /// `StarNodes` exist in a lattice and correspond to a star generated from a prefix of the network along with other calculated properties.
    ///
    pub struct StarNode<D: Dimension> {
        star: Star<D>,
        star_cdf: Option<NNVFloat>,
        cdf_delta: NNVFloat,
        axis_aligned_input_bounds: Option<Bounds1>,
        output_bounds: Option<(NNVFloat, NNVFloat)>,
        gaussian_distribution: Option<GaussianDistribution>,
    }
    #[automatically_derived]
    #[allow(unused_qualifications)]
    impl<D: ::core::fmt::Debug + Dimension> ::core::fmt::Debug for StarNode<D> {
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            match *self {
                StarNode {
                    star: ref __self_0_0,
                    star_cdf: ref __self_0_1,
                    cdf_delta: ref __self_0_2,
                    axis_aligned_input_bounds: ref __self_0_3,
                    output_bounds: ref __self_0_4,
                    gaussian_distribution: ref __self_0_5,
                } => {
                    let debug_trait_builder =
                        &mut ::core::fmt::Formatter::debug_struct(f, "StarNode");
                    let _ = ::core::fmt::DebugStruct::field(
                        debug_trait_builder,
                        "star",
                        &&(*__self_0_0),
                    );
                    let _ = ::core::fmt::DebugStruct::field(
                        debug_trait_builder,
                        "star_cdf",
                        &&(*__self_0_1),
                    );
                    let _ = ::core::fmt::DebugStruct::field(
                        debug_trait_builder,
                        "cdf_delta",
                        &&(*__self_0_2),
                    );
                    let _ = ::core::fmt::DebugStruct::field(
                        debug_trait_builder,
                        "axis_aligned_input_bounds",
                        &&(*__self_0_3),
                    );
                    let _ = ::core::fmt::DebugStruct::field(
                        debug_trait_builder,
                        "output_bounds",
                        &&(*__self_0_4),
                    );
                    let _ = ::core::fmt::DebugStruct::field(
                        debug_trait_builder,
                        "gaussian_distribution",
                        &&(*__self_0_5),
                    );
                    ::core::fmt::DebugStruct::finish(debug_trait_builder)
                }
            }
        }
    }
    #[automatically_derived]
    #[allow(unused_qualifications)]
    impl<D: ::core::clone::Clone + Dimension> ::core::clone::Clone for StarNode<D> {
        #[inline]
        fn clone(&self) -> StarNode<D> {
            match *self {
                StarNode {
                    star: ref __self_0_0,
                    star_cdf: ref __self_0_1,
                    cdf_delta: ref __self_0_2,
                    axis_aligned_input_bounds: ref __self_0_3,
                    output_bounds: ref __self_0_4,
                    gaussian_distribution: ref __self_0_5,
                } => StarNode {
                    star: ::core::clone::Clone::clone(&(*__self_0_0)),
                    star_cdf: ::core::clone::Clone::clone(&(*__self_0_1)),
                    cdf_delta: ::core::clone::Clone::clone(&(*__self_0_2)),
                    axis_aligned_input_bounds: ::core::clone::Clone::clone(&(*__self_0_3)),
                    output_bounds: ::core::clone::Clone::clone(&(*__self_0_4)),
                    gaussian_distribution: ::core::clone::Clone::clone(&(*__self_0_5)),
                },
            }
        }
    }
    #[doc(hidden)]
    #[allow(non_upper_case_globals, unused_attributes, unused_qualifications)]
    const _: () = {
        #[allow(unused_extern_crates, clippy::useless_attribute)]
        extern crate serde as _serde;
        #[automatically_derived]
        impl<'de, D: Dimension> _serde::Deserialize<'de> for StarNode<D>
        where
            D: _serde::Deserialize<'de>,
        {
            fn deserialize<__D>(__deserializer: __D) -> _serde::__private::Result<Self, __D::Error>
            where
                __D: _serde::Deserializer<'de>,
            {
                #[allow(non_camel_case_types)]
                enum __Field {
                    __field0,
                    __field1,
                    __field2,
                    __field3,
                    __field4,
                    __field5,
                    __ignore,
                }
                struct __FieldVisitor;
                impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                    type Value = __Field;
                    fn expecting(
                        &self,
                        __formatter: &mut _serde::__private::Formatter,
                    ) -> _serde::__private::fmt::Result {
                        _serde::__private::Formatter::write_str(__formatter, "field identifier")
                    }
                    fn visit_u64<__E>(
                        self,
                        __value: u64,
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            0u64 => _serde::__private::Ok(__Field::__field0),
                            1u64 => _serde::__private::Ok(__Field::__field1),
                            2u64 => _serde::__private::Ok(__Field::__field2),
                            3u64 => _serde::__private::Ok(__Field::__field3),
                            4u64 => _serde::__private::Ok(__Field::__field4),
                            5u64 => _serde::__private::Ok(__Field::__field5),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                    fn visit_str<__E>(
                        self,
                        __value: &str,
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            "star" => _serde::__private::Ok(__Field::__field0),
                            "star_cdf" => _serde::__private::Ok(__Field::__field1),
                            "cdf_delta" => _serde::__private::Ok(__Field::__field2),
                            "axis_aligned_input_bounds" => _serde::__private::Ok(__Field::__field3),
                            "output_bounds" => _serde::__private::Ok(__Field::__field4),
                            "gaussian_distribution" => _serde::__private::Ok(__Field::__field5),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                    fn visit_bytes<__E>(
                        self,
                        __value: &[u8],
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            b"star" => _serde::__private::Ok(__Field::__field0),
                            b"star_cdf" => _serde::__private::Ok(__Field::__field1),
                            b"cdf_delta" => _serde::__private::Ok(__Field::__field2),
                            b"axis_aligned_input_bounds" => {
                                _serde::__private::Ok(__Field::__field3)
                            }
                            b"output_bounds" => _serde::__private::Ok(__Field::__field4),
                            b"gaussian_distribution" => _serde::__private::Ok(__Field::__field5),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                }
                impl<'de> _serde::Deserialize<'de> for __Field {
                    #[inline]
                    fn deserialize<__D>(
                        __deserializer: __D,
                    ) -> _serde::__private::Result<Self, __D::Error>
                    where
                        __D: _serde::Deserializer<'de>,
                    {
                        _serde::Deserializer::deserialize_identifier(__deserializer, __FieldVisitor)
                    }
                }
                struct __Visitor<'de, D: Dimension>
                where
                    D: _serde::Deserialize<'de>,
                {
                    marker: _serde::__private::PhantomData<StarNode<D>>,
                    lifetime: _serde::__private::PhantomData<&'de ()>,
                }
                impl<'de, D: Dimension> _serde::de::Visitor<'de> for __Visitor<'de, D>
                where
                    D: _serde::Deserialize<'de>,
                {
                    type Value = StarNode<D>;
                    fn expecting(
                        &self,
                        __formatter: &mut _serde::__private::Formatter,
                    ) -> _serde::__private::fmt::Result {
                        _serde::__private::Formatter::write_str(__formatter, "struct StarNode")
                    }
                    #[inline]
                    fn visit_seq<__A>(
                        self,
                        mut __seq: __A,
                    ) -> _serde::__private::Result<Self::Value, __A::Error>
                    where
                        __A: _serde::de::SeqAccess<'de>,
                    {
                        let __field0 = match match _serde::de::SeqAccess::next_element::<Star<D>>(
                            &mut __seq,
                        ) {
                            _serde::__private::Ok(__val) => __val,
                            _serde::__private::Err(__err) => {
                                return _serde::__private::Err(__err);
                            }
                        } {
                            _serde::__private::Some(__value) => __value,
                            _serde::__private::None => {
                                return _serde::__private::Err(_serde::de::Error::invalid_length(
                                    0usize,
                                    &"struct StarNode with 6 elements",
                                ));
                            }
                        };
                        let __field1 = match match _serde::de::SeqAccess::next_element::<
                            Option<NNVFloat>,
                        >(&mut __seq)
                        {
                            _serde::__private::Ok(__val) => __val,
                            _serde::__private::Err(__err) => {
                                return _serde::__private::Err(__err);
                            }
                        } {
                            _serde::__private::Some(__value) => __value,
                            _serde::__private::None => {
                                return _serde::__private::Err(_serde::de::Error::invalid_length(
                                    1usize,
                                    &"struct StarNode with 6 elements",
                                ));
                            }
                        };
                        let __field2 =
                            match match _serde::de::SeqAccess::next_element::<NNVFloat>(&mut __seq)
                            {
                                _serde::__private::Ok(__val) => __val,
                                _serde::__private::Err(__err) => {
                                    return _serde::__private::Err(__err);
                                }
                            } {
                                _serde::__private::Some(__value) => __value,
                                _serde::__private::None => {
                                    return _serde::__private::Err(
                                        _serde::de::Error::invalid_length(
                                            2usize,
                                            &"struct StarNode with 6 elements",
                                        ),
                                    );
                                }
                            };
                        let __field3 = match match _serde::de::SeqAccess::next_element::<
                            Option<Bounds1>,
                        >(&mut __seq)
                        {
                            _serde::__private::Ok(__val) => __val,
                            _serde::__private::Err(__err) => {
                                return _serde::__private::Err(__err);
                            }
                        } {
                            _serde::__private::Some(__value) => __value,
                            _serde::__private::None => {
                                return _serde::__private::Err(_serde::de::Error::invalid_length(
                                    3usize,
                                    &"struct StarNode with 6 elements",
                                ));
                            }
                        };
                        let __field4 = match match _serde::de::SeqAccess::next_element::<
                            Option<(NNVFloat, NNVFloat)>,
                        >(&mut __seq)
                        {
                            _serde::__private::Ok(__val) => __val,
                            _serde::__private::Err(__err) => {
                                return _serde::__private::Err(__err);
                            }
                        } {
                            _serde::__private::Some(__value) => __value,
                            _serde::__private::None => {
                                return _serde::__private::Err(_serde::de::Error::invalid_length(
                                    4usize,
                                    &"struct StarNode with 6 elements",
                                ));
                            }
                        };
                        let __field5 = match match _serde::de::SeqAccess::next_element::<
                            Option<GaussianDistribution>,
                        >(&mut __seq)
                        {
                            _serde::__private::Ok(__val) => __val,
                            _serde::__private::Err(__err) => {
                                return _serde::__private::Err(__err);
                            }
                        } {
                            _serde::__private::Some(__value) => __value,
                            _serde::__private::None => {
                                return _serde::__private::Err(_serde::de::Error::invalid_length(
                                    5usize,
                                    &"struct StarNode with 6 elements",
                                ));
                            }
                        };
                        _serde::__private::Ok(StarNode {
                            star: __field0,
                            star_cdf: __field1,
                            cdf_delta: __field2,
                            axis_aligned_input_bounds: __field3,
                            output_bounds: __field4,
                            gaussian_distribution: __field5,
                        })
                    }
                    #[inline]
                    fn visit_map<__A>(
                        self,
                        mut __map: __A,
                    ) -> _serde::__private::Result<Self::Value, __A::Error>
                    where
                        __A: _serde::de::MapAccess<'de>,
                    {
                        let mut __field0: _serde::__private::Option<Star<D>> =
                            _serde::__private::None;
                        let mut __field1: _serde::__private::Option<Option<NNVFloat>> =
                            _serde::__private::None;
                        let mut __field2: _serde::__private::Option<NNVFloat> =
                            _serde::__private::None;
                        let mut __field3: _serde::__private::Option<Option<Bounds1>> =
                            _serde::__private::None;
                        let mut __field4: _serde::__private::Option<Option<(NNVFloat, NNVFloat)>> =
                            _serde::__private::None;
                        let mut __field5: _serde::__private::Option<Option<GaussianDistribution>> =
                            _serde::__private::None;
                        while let _serde::__private::Some(__key) =
                            match _serde::de::MapAccess::next_key::<__Field>(&mut __map) {
                                _serde::__private::Ok(__val) => __val,
                                _serde::__private::Err(__err) => {
                                    return _serde::__private::Err(__err);
                                }
                            }
                        {
                            match __key {
                                __Field::__field0 => {
                                    if _serde::__private::Option::is_some(&__field0) {
                                        return _serde::__private::Err(
                                            <__A::Error as _serde::de::Error>::duplicate_field(
                                                "star",
                                            ),
                                        );
                                    }
                                    __field0 = _serde::__private::Some(
                                        match _serde::de::MapAccess::next_value::<Star<D>>(
                                            &mut __map,
                                        ) {
                                            _serde::__private::Ok(__val) => __val,
                                            _serde::__private::Err(__err) => {
                                                return _serde::__private::Err(__err);
                                            }
                                        },
                                    );
                                }
                                __Field::__field1 => {
                                    if _serde::__private::Option::is_some(&__field1) {
                                        return _serde::__private::Err(
                                            <__A::Error as _serde::de::Error>::duplicate_field(
                                                "star_cdf",
                                            ),
                                        );
                                    }
                                    __field1 = _serde::__private::Some(
                                        match _serde::de::MapAccess::next_value::<Option<NNVFloat>>(
                                            &mut __map,
                                        ) {
                                            _serde::__private::Ok(__val) => __val,
                                            _serde::__private::Err(__err) => {
                                                return _serde::__private::Err(__err);
                                            }
                                        },
                                    );
                                }
                                __Field::__field2 => {
                                    if _serde::__private::Option::is_some(&__field2) {
                                        return _serde::__private::Err(
                                            <__A::Error as _serde::de::Error>::duplicate_field(
                                                "cdf_delta",
                                            ),
                                        );
                                    }
                                    __field2 = _serde::__private::Some(
                                        match _serde::de::MapAccess::next_value::<NNVFloat>(
                                            &mut __map,
                                        ) {
                                            _serde::__private::Ok(__val) => __val,
                                            _serde::__private::Err(__err) => {
                                                return _serde::__private::Err(__err);
                                            }
                                        },
                                    );
                                }
                                __Field::__field3 => {
                                    if _serde::__private::Option::is_some(&__field3) {
                                        return _serde::__private::Err(
                                            <__A::Error as _serde::de::Error>::duplicate_field(
                                                "axis_aligned_input_bounds",
                                            ),
                                        );
                                    }
                                    __field3 = _serde::__private::Some(
                                        match _serde::de::MapAccess::next_value::<Option<Bounds1>>(
                                            &mut __map,
                                        ) {
                                            _serde::__private::Ok(__val) => __val,
                                            _serde::__private::Err(__err) => {
                                                return _serde::__private::Err(__err);
                                            }
                                        },
                                    );
                                }
                                __Field::__field4 => {
                                    if _serde::__private::Option::is_some(&__field4) {
                                        return _serde::__private::Err(
                                            <__A::Error as _serde::de::Error>::duplicate_field(
                                                "output_bounds",
                                            ),
                                        );
                                    }
                                    __field4 = _serde::__private::Some(
                                        match _serde::de::MapAccess::next_value::<
                                            Option<(NNVFloat, NNVFloat)>,
                                        >(&mut __map)
                                        {
                                            _serde::__private::Ok(__val) => __val,
                                            _serde::__private::Err(__err) => {
                                                return _serde::__private::Err(__err);
                                            }
                                        },
                                    );
                                }
                                __Field::__field5 => {
                                    if _serde::__private::Option::is_some(&__field5) {
                                        return _serde::__private::Err(
                                            <__A::Error as _serde::de::Error>::duplicate_field(
                                                "gaussian_distribution",
                                            ),
                                        );
                                    }
                                    __field5 = _serde::__private::Some(
                                        match _serde::de::MapAccess::next_value::<
                                            Option<GaussianDistribution>,
                                        >(&mut __map)
                                        {
                                            _serde::__private::Ok(__val) => __val,
                                            _serde::__private::Err(__err) => {
                                                return _serde::__private::Err(__err);
                                            }
                                        },
                                    );
                                }
                                _ => {
                                    let _ = match _serde::de::MapAccess::next_value::<
                                        _serde::de::IgnoredAny,
                                    >(&mut __map)
                                    {
                                        _serde::__private::Ok(__val) => __val,
                                        _serde::__private::Err(__err) => {
                                            return _serde::__private::Err(__err);
                                        }
                                    };
                                }
                            }
                        }
                        let __field0 = match __field0 {
                            _serde::__private::Some(__field0) => __field0,
                            _serde::__private::None => {
                                match _serde::__private::de::missing_field("star") {
                                    _serde::__private::Ok(__val) => __val,
                                    _serde::__private::Err(__err) => {
                                        return _serde::__private::Err(__err);
                                    }
                                }
                            }
                        };
                        let __field1 = match __field1 {
                            _serde::__private::Some(__field1) => __field1,
                            _serde::__private::None => {
                                match _serde::__private::de::missing_field("star_cdf") {
                                    _serde::__private::Ok(__val) => __val,
                                    _serde::__private::Err(__err) => {
                                        return _serde::__private::Err(__err);
                                    }
                                }
                            }
                        };
                        let __field2 = match __field2 {
                            _serde::__private::Some(__field2) => __field2,
                            _serde::__private::None => {
                                match _serde::__private::de::missing_field("cdf_delta") {
                                    _serde::__private::Ok(__val) => __val,
                                    _serde::__private::Err(__err) => {
                                        return _serde::__private::Err(__err);
                                    }
                                }
                            }
                        };
                        let __field3 = match __field3 {
                            _serde::__private::Some(__field3) => __field3,
                            _serde::__private::None => match _serde::__private::de::missing_field(
                                "axis_aligned_input_bounds",
                            ) {
                                _serde::__private::Ok(__val) => __val,
                                _serde::__private::Err(__err) => {
                                    return _serde::__private::Err(__err);
                                }
                            },
                        };
                        let __field4 = match __field4 {
                            _serde::__private::Some(__field4) => __field4,
                            _serde::__private::None => {
                                match _serde::__private::de::missing_field("output_bounds") {
                                    _serde::__private::Ok(__val) => __val,
                                    _serde::__private::Err(__err) => {
                                        return _serde::__private::Err(__err);
                                    }
                                }
                            }
                        };
                        let __field5 = match __field5 {
                            _serde::__private::Some(__field5) => __field5,
                            _serde::__private::None => {
                                match _serde::__private::de::missing_field("gaussian_distribution")
                                {
                                    _serde::__private::Ok(__val) => __val,
                                    _serde::__private::Err(__err) => {
                                        return _serde::__private::Err(__err);
                                    }
                                }
                            }
                        };
                        _serde::__private::Ok(StarNode {
                            star: __field0,
                            star_cdf: __field1,
                            cdf_delta: __field2,
                            axis_aligned_input_bounds: __field3,
                            output_bounds: __field4,
                            gaussian_distribution: __field5,
                        })
                    }
                }
                const FIELDS: &'static [&'static str] = &[
                    "star",
                    "star_cdf",
                    "cdf_delta",
                    "axis_aligned_input_bounds",
                    "output_bounds",
                    "gaussian_distribution",
                ];
                _serde::Deserializer::deserialize_struct(
                    __deserializer,
                    "StarNode",
                    FIELDS,
                    __Visitor {
                        marker: _serde::__private::PhantomData::<StarNode<D>>,
                        lifetime: _serde::__private::PhantomData,
                    },
                )
            }
        }
    };
    #[doc(hidden)]
    #[allow(non_upper_case_globals, unused_attributes, unused_qualifications)]
    const _: () = {
        #[allow(unused_extern_crates, clippy::useless_attribute)]
        extern crate serde as _serde;
        #[automatically_derived]
        impl<D: Dimension> _serde::Serialize for StarNode<D>
        where
            D: _serde::Serialize,
        {
            fn serialize<__S>(
                &self,
                __serializer: __S,
            ) -> _serde::__private::Result<__S::Ok, __S::Error>
            where
                __S: _serde::Serializer,
            {
                let mut __serde_state = match _serde::Serializer::serialize_struct(
                    __serializer,
                    "StarNode",
                    false as usize + 1 + 1 + 1 + 1 + 1 + 1,
                ) {
                    _serde::__private::Ok(__val) => __val,
                    _serde::__private::Err(__err) => {
                        return _serde::__private::Err(__err);
                    }
                };
                match _serde::ser::SerializeStruct::serialize_field(
                    &mut __serde_state,
                    "star",
                    &self.star,
                ) {
                    _serde::__private::Ok(__val) => __val,
                    _serde::__private::Err(__err) => {
                        return _serde::__private::Err(__err);
                    }
                };
                match _serde::ser::SerializeStruct::serialize_field(
                    &mut __serde_state,
                    "star_cdf",
                    &self.star_cdf,
                ) {
                    _serde::__private::Ok(__val) => __val,
                    _serde::__private::Err(__err) => {
                        return _serde::__private::Err(__err);
                    }
                };
                match _serde::ser::SerializeStruct::serialize_field(
                    &mut __serde_state,
                    "cdf_delta",
                    &self.cdf_delta,
                ) {
                    _serde::__private::Ok(__val) => __val,
                    _serde::__private::Err(__err) => {
                        return _serde::__private::Err(__err);
                    }
                };
                match _serde::ser::SerializeStruct::serialize_field(
                    &mut __serde_state,
                    "axis_aligned_input_bounds",
                    &self.axis_aligned_input_bounds,
                ) {
                    _serde::__private::Ok(__val) => __val,
                    _serde::__private::Err(__err) => {
                        return _serde::__private::Err(__err);
                    }
                };
                match _serde::ser::SerializeStruct::serialize_field(
                    &mut __serde_state,
                    "output_bounds",
                    &self.output_bounds,
                ) {
                    _serde::__private::Ok(__val) => __val,
                    _serde::__private::Err(__err) => {
                        return _serde::__private::Err(__err);
                    }
                };
                match _serde::ser::SerializeStruct::serialize_field(
                    &mut __serde_state,
                    "gaussian_distribution",
                    &self.gaussian_distribution,
                ) {
                    _serde::__private::Ok(__val) => __val,
                    _serde::__private::Err(__err) => {
                        return _serde::__private::Err(__err);
                    }
                };
                _serde::ser::SerializeStruct::end(__serde_state)
            }
        }
    };
    impl<D: Dimension> StarNode<D> {
        pub fn default(star: Star<D>, axis_aligned_input_bounds: Option<Bounds1>) -> Self {
            Self {
                star,
                star_cdf: None,
                cdf_delta: 0.,
                axis_aligned_input_bounds,
                output_bounds: None,
                gaussian_distribution: None,
            }
        }
        pub fn get_star(&self) -> &Star<D> {
            &self.star
        }
        pub fn try_get_cdf(&self) -> Option<NNVFloat> {
            self.star_cdf
        }
        pub fn set_cdf(&mut self, val: NNVFloat) {
            self.star_cdf = Some(val);
        }
        pub fn reset_cdf(&mut self) {
            self.star_cdf = None;
            self.cdf_delta = 0.;
        }
        /// # Panics
        pub fn add_cdf(&mut self, add: NNVFloat) {
            self.cdf_delta += add;
        }
        pub fn try_get_output_bounds(&self) -> Option<(NNVFloat, NNVFloat)> {
            self.output_bounds
        }
        pub fn set_output_bounds(&mut self, val: (NNVFloat, NNVFloat)) {
            self.output_bounds = Some(val);
        }
    }
    impl StarNode<Ix2> {
        pub fn is_input_member(&self, point: &ArrayView1<NNVFloat>) -> bool {
            match self.star.input_space_polytope() {
                Some(poly) => poly.is_member(point),
                None => true,
            }
        }
        pub fn get_reduced_input_polytope(
            &self,
            bounds: &Option<Vec<Bounds1>>,
        ) -> Option<Polytope> {
            self.star
                .input_space_polytope()
                .and_then(|x| x.reduce_fixed_inputs(bounds))
        }
        /// None indicates that the distribution hasn't been calculated/constructed
        pub const fn try_get_gaussian_distribution(&self) -> Option<&GaussianDistribution> {
            self.gaussian_distribution.as_ref()
        }
        pub fn set_gaussian_distribution(&mut self, val: GaussianDistribution) {
            self.gaussian_distribution = Some(val);
        }
        /// # Panics
        pub fn get_gaussian_distribution(
            &mut self,
            loc: ArrayView1<NNVFloat>,
            scale: ArrayView2<NNVFloat>,
            max_accept_reject_iters: usize,
            stability_eps: NNVFloat,
            input_bounds_opt: &Option<Bounds1>,
        ) -> &mut GaussianDistribution {
            if self.gaussian_distribution.is_none() {
                self.gaussian_distribution = self.star.get_input_trunc_gaussian(
                    loc,
                    scale,
                    max_accept_reject_iters,
                    stability_eps,
                    input_bounds_opt,
                );
                if self.gaussian_distribution.is_none() {
                    self.gaussian_distribution = Some(GaussianDistribution::Gaussian {
                        loc: loc.to_owned(),
                        scale: scale.diag().to_owned(),
                    });
                }
            }
            self.gaussian_distribution.as_mut().unwrap()
        }
        pub fn forward(&self, x: &Array1<NNVFloat>) -> Array1<NNVFloat> {
            self.star.get_representation().apply(&x.view())
        }
        #[must_use]
        pub fn get_unsafe_star(&self, safe_value: NNVFloat) -> Self {
            let safe_star = self.star.get_safe_subset(safe_value);
            Self {
                star: safe_star,
                star_cdf: None,
                cdf_delta: 0.,
                axis_aligned_input_bounds: None,
                output_bounds: None,
                gaussian_distribution: None,
            }
        }
        #[must_use]
        pub fn get_safe_star(&self, safe_value: NNVFloat) -> Self {
            let safe_star = self.star.get_safe_subset(safe_value);
            Self {
                star: safe_star,
                star_cdf: None,
                cdf_delta: 0.,
                axis_aligned_input_bounds: None,
                output_bounds: None,
                gaussian_distribution: None,
            }
        }
        pub fn gaussian_cdf<R: Rng>(
            &mut self,
            mu: ArrayView1<NNVFloat>,
            sigma: ArrayView2<NNVFloat>,
            n: usize,
            max_iters: usize,
            rng: &mut R,
            stability_eps: NNVFloat,
            input_bounds_opt: &Option<Bounds1>,
        ) -> NNVFloat {
            let cdf = self.star_cdf.unwrap_or_else(|| {
                let cdf: NNVFloat = self
                    .get_gaussian_distribution(
                        mu,
                        sigma,
                        max_iters,
                        stability_eps,
                        input_bounds_opt,
                    )
                    .cdf(n, rng);
                if true {
                    if !cdf.is_sign_positive() {
                        ::core::panicking::panic("assertion failed: cdf.is_sign_positive()")
                    };
                };
                self.star_cdf = Some(cdf);
                cdf
            });
            let cdf_sum = cdf + self.cdf_delta;
            if cdf_sum.is_sign_negative() {
                NNVFloat::epsilon()
            } else {
                cdf_sum
            }
        }
        /// # Panics
        pub fn gaussian_sample<R: Rng>(
            &mut self,
            rng: &mut R,
            mu: ArrayView1<NNVFloat>,
            sigma: ArrayView2<NNVFloat>,
            n: usize,
            max_iters: usize,
            tilting_initialization: Option<&TiltingSolution>,
            stability_eps: NNVFloat,
            input_bounds_opt: &Option<Bounds1>,
        ) -> Vec<Array1<NNVFloat>> {
            let distribution = self.get_gaussian_distribution(
                mu,
                sigma,
                max_iters,
                stability_eps,
                input_bounds_opt,
            );
            distribution.populate_tilting_solution(tilting_initialization);
            distribution.sample_n(n, rng)
        }
        pub const fn try_get_axis_aligned_input_bounds(&self) -> &Option<Bounds1> {
            &self.axis_aligned_input_bounds
        }
        /// # Panics
        pub fn get_axis_aligned_input_bounds(&mut self, outer_bounds: &Bounds1) -> &Bounds1 {
            if self.axis_aligned_input_bounds.is_none() {
                self.axis_aligned_input_bounds = Some(
                    self.star
                        .calculate_output_axis_aligned_bounding_box(outer_bounds),
                );
            }
            self.axis_aligned_input_bounds.as_ref().unwrap()
        }
    }
}
pub mod starsets {
    mod new_graph_starset {
        use std::cell::{Ref, RefCell};
        use super::new_starset::{StarId, StarRelationship, StarRelationshipId, StarSet, StarSet2};
        use crate::bounds::{Bounds, Bounds1};
        use crate::dnn::DNN;
        use crate::graph::{Graph, RepresentationId};
        use crate::star::Star;
        use ndarray::Dimension;
        use ndarray::Ix2;
        pub struct GraphStarset<D: 'static + Dimension> {
            /// The network for which the starset is generated
            dnn: DNN,
            /// Storage structure for stars
            arena: RefCell<Vec<Star<D>>>,
            /// The RepresentationId that each star represents
            representations: RefCell<Vec<RepresentationId>>,
            /// Axis aligned input bounds for each star
            input_bounds: RefCell<Vec<Bounds<D>>>,
            /// The relationships between stars, includes the associated graph operation
            relationships: RefCell<Vec<StarRelationship>>,
        }
        impl<D: Dimension> GraphStarset<D> {
            pub fn new(dnn: DNN, input_star: Star<D>, input_bounds: Bounds<D>) -> Self {
                Self {
                    dnn,
                    arena: RefCell::new(<[_]>::into_vec(box [input_star])),
                    representations: RefCell::new(<[_]>::into_vec(box [RepresentationId::new(
                        0, None,
                    )])),
                    input_bounds: RefCell::new(<[_]>::into_vec(box [input_bounds])),
                    relationships: RefCell::new(::alloc::vec::Vec::new()),
                }
            }
        }
        impl<D: 'static + Dimension> StarSet<D> for GraphStarset<D> {
            fn get_graph(&self) -> &Graph {
                self.dnn.get_graph()
            }
            fn get_dnn(&self) -> &DNN {
                &self.dnn
            }
            fn get_root_id(&self) -> StarId {
                0
            }
            fn get_star_representation_id(&self, star_id: usize) -> RepresentationId {
                self.representations.borrow()[star_id]
            }
            fn get_star(&self, star_id: StarId) -> Ref<Star<D>> {
                if !(star_id < self.arena.borrow().len()) {
                    ::core::panicking::panic(
                        "assertion failed: star_id < self.arena.borrow().len()",
                    )
                };
                Ref::map(self.arena.borrow(), |vec| &vec[star_id])
            }
            fn get_relationship(
                &self,
                relationship_id: StarRelationshipId,
            ) -> Ref<StarRelationship> {
                if !(relationship_id < self.relationships.borrow().len()) {
                    ::core::panicking::panic(
                        "assertion failed: relationship_id < self.relationships.borrow().len()",
                    )
                };
                Ref::map(self.relationships.borrow(), |vec| &vec[relationship_id])
            }
            fn add_star(
                &self,
                star: Star<D>,
                representation_id: RepresentationId,
                axis_aligned_input_bounds: Bounds<D>,
            ) -> StarId {
                let star_id = self.arena.borrow().len();
                self.arena.borrow_mut().push(star);
                self.representations.borrow_mut().push(representation_id);
                self.input_bounds
                    .borrow_mut()
                    .push(axis_aligned_input_bounds);
                match (
                    &self.arena.borrow().len(),
                    &self.representations.borrow().len(),
                ) {
                    (left_val, right_val) => {
                        if !(*left_val == *right_val) {
                            let kind = ::core::panicking::AssertKind::Eq;
                            ::core::panicking::assert_failed(
                                kind,
                                &*left_val,
                                &*right_val,
                                ::core::option::Option::None,
                            );
                        }
                    }
                };
                star_id
            }
            fn add_relationship(
                &self,
                star_rel: super::new_starset::StarRelationship,
            ) -> StarRelationshipId {
                let rel_id = self.relationships.borrow().len();
                self.relationships.borrow_mut().push(star_rel);
                rel_id
            }
        }
        impl StarSet2 for GraphStarset<Ix2> {
            fn get_input_dim(&self) -> usize {
                self.arena.borrow()[0].input_dim()
            }
            /// TODO: Implement with a cache because it is expensive
            fn get_axis_aligned_input_bounds(&self, star_id: StarId) -> &Bounds1 {
                if !(star_id < self.input_bounds.borrow().len()) {
                    ::core::panicking::panic(
                        "assertion failed: star_id < self.input_bounds.borrow().len()",
                    )
                };
                &self.input_bounds.borrow()[star_id]
            }
        }
    }
    mod new_starset {
        use std::cell::Cell;
        use std::cell::Ref;
        use std::ops::Deref;
        use std::rc::Rc;
        use crate::bounds::Bounds;
        use crate::bounds::Bounds1;
        use crate::dnn::DNN;
        use crate::graph::Graph;
        use crate::graph::OperationId;
        use crate::graph::RepresentationId;
        use crate::graph::StarOperation;
        use crate::star::Star;
        use ndarray::Dimension;
        use ndarray::Ix2;
        use serde::{Deserialize, Serialize};
        pub type StarId = usize;
        pub type StarRelationshipId = usize;
        pub struct StarRelationship {
            pub operation_id: OperationId,
            pub step: Option<usize>,
            pub input_star_ids: Vec<usize>,
            pub output_star_ids: Vec<usize>,
        }
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl ::core::fmt::Debug for StarRelationship {
            fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
                match *self {
                    StarRelationship {
                        operation_id: ref __self_0_0,
                        step: ref __self_0_1,
                        input_star_ids: ref __self_0_2,
                        output_star_ids: ref __self_0_3,
                    } => {
                        let debug_trait_builder =
                            &mut ::core::fmt::Formatter::debug_struct(f, "StarRelationship");
                        let _ = ::core::fmt::DebugStruct::field(
                            debug_trait_builder,
                            "operation_id",
                            &&(*__self_0_0),
                        );
                        let _ = ::core::fmt::DebugStruct::field(
                            debug_trait_builder,
                            "step",
                            &&(*__self_0_1),
                        );
                        let _ = ::core::fmt::DebugStruct::field(
                            debug_trait_builder,
                            "input_star_ids",
                            &&(*__self_0_2),
                        );
                        let _ = ::core::fmt::DebugStruct::field(
                            debug_trait_builder,
                            "output_star_ids",
                            &&(*__self_0_3),
                        );
                        ::core::fmt::DebugStruct::finish(debug_trait_builder)
                    }
                }
            }
        }
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl ::core::clone::Clone for StarRelationship {
            #[inline]
            fn clone(&self) -> StarRelationship {
                match *self {
                    StarRelationship {
                        operation_id: ref __self_0_0,
                        step: ref __self_0_1,
                        input_star_ids: ref __self_0_2,
                        output_star_ids: ref __self_0_3,
                    } => StarRelationship {
                        operation_id: ::core::clone::Clone::clone(&(*__self_0_0)),
                        step: ::core::clone::Clone::clone(&(*__self_0_1)),
                        input_star_ids: ::core::clone::Clone::clone(&(*__self_0_2)),
                        output_star_ids: ::core::clone::Clone::clone(&(*__self_0_3)),
                    },
                }
            }
        }
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl ::core::default::Default for StarRelationship {
            #[inline]
            fn default() -> StarRelationship {
                StarRelationship {
                    operation_id: ::core::default::Default::default(),
                    step: ::core::default::Default::default(),
                    input_star_ids: ::core::default::Default::default(),
                    output_star_ids: ::core::default::Default::default(),
                }
            }
        }
        #[doc(hidden)]
        #[allow(non_upper_case_globals, unused_attributes, unused_qualifications)]
        const _: () = {
            #[allow(unused_extern_crates, clippy::useless_attribute)]
            extern crate serde as _serde;
            #[automatically_derived]
            impl _serde::Serialize for StarRelationship {
                fn serialize<__S>(
                    &self,
                    __serializer: __S,
                ) -> _serde::__private::Result<__S::Ok, __S::Error>
                where
                    __S: _serde::Serializer,
                {
                    let mut __serde_state = match _serde::Serializer::serialize_struct(
                        __serializer,
                        "StarRelationship",
                        false as usize + 1 + 1 + 1 + 1,
                    ) {
                        _serde::__private::Ok(__val) => __val,
                        _serde::__private::Err(__err) => {
                            return _serde::__private::Err(__err);
                        }
                    };
                    match _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "operation_id",
                        &self.operation_id,
                    ) {
                        _serde::__private::Ok(__val) => __val,
                        _serde::__private::Err(__err) => {
                            return _serde::__private::Err(__err);
                        }
                    };
                    match _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "step",
                        &self.step,
                    ) {
                        _serde::__private::Ok(__val) => __val,
                        _serde::__private::Err(__err) => {
                            return _serde::__private::Err(__err);
                        }
                    };
                    match _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "input_star_ids",
                        &self.input_star_ids,
                    ) {
                        _serde::__private::Ok(__val) => __val,
                        _serde::__private::Err(__err) => {
                            return _serde::__private::Err(__err);
                        }
                    };
                    match _serde::ser::SerializeStruct::serialize_field(
                        &mut __serde_state,
                        "output_star_ids",
                        &self.output_star_ids,
                    ) {
                        _serde::__private::Ok(__val) => __val,
                        _serde::__private::Err(__err) => {
                            return _serde::__private::Err(__err);
                        }
                    };
                    _serde::ser::SerializeStruct::end(__serde_state)
                }
            }
        };
        #[doc(hidden)]
        #[allow(non_upper_case_globals, unused_attributes, unused_qualifications)]
        const _: () = {
            #[allow(unused_extern_crates, clippy::useless_attribute)]
            extern crate serde as _serde;
            #[automatically_derived]
            impl<'de> _serde::Deserialize<'de> for StarRelationship {
                fn deserialize<__D>(
                    __deserializer: __D,
                ) -> _serde::__private::Result<Self, __D::Error>
                where
                    __D: _serde::Deserializer<'de>,
                {
                    #[allow(non_camel_case_types)]
                    enum __Field {
                        __field0,
                        __field1,
                        __field2,
                        __field3,
                        __ignore,
                    }
                    struct __FieldVisitor;
                    impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                        type Value = __Field;
                        fn expecting(
                            &self,
                            __formatter: &mut _serde::__private::Formatter,
                        ) -> _serde::__private::fmt::Result {
                            _serde::__private::Formatter::write_str(__formatter, "field identifier")
                        }
                        fn visit_u64<__E>(
                            self,
                            __value: u64,
                        ) -> _serde::__private::Result<Self::Value, __E>
                        where
                            __E: _serde::de::Error,
                        {
                            match __value {
                                0u64 => _serde::__private::Ok(__Field::__field0),
                                1u64 => _serde::__private::Ok(__Field::__field1),
                                2u64 => _serde::__private::Ok(__Field::__field2),
                                3u64 => _serde::__private::Ok(__Field::__field3),
                                _ => _serde::__private::Ok(__Field::__ignore),
                            }
                        }
                        fn visit_str<__E>(
                            self,
                            __value: &str,
                        ) -> _serde::__private::Result<Self::Value, __E>
                        where
                            __E: _serde::de::Error,
                        {
                            match __value {
                                "operation_id" => _serde::__private::Ok(__Field::__field0),
                                "step" => _serde::__private::Ok(__Field::__field1),
                                "input_star_ids" => _serde::__private::Ok(__Field::__field2),
                                "output_star_ids" => _serde::__private::Ok(__Field::__field3),
                                _ => _serde::__private::Ok(__Field::__ignore),
                            }
                        }
                        fn visit_bytes<__E>(
                            self,
                            __value: &[u8],
                        ) -> _serde::__private::Result<Self::Value, __E>
                        where
                            __E: _serde::de::Error,
                        {
                            match __value {
                                b"operation_id" => _serde::__private::Ok(__Field::__field0),
                                b"step" => _serde::__private::Ok(__Field::__field1),
                                b"input_star_ids" => _serde::__private::Ok(__Field::__field2),
                                b"output_star_ids" => _serde::__private::Ok(__Field::__field3),
                                _ => _serde::__private::Ok(__Field::__ignore),
                            }
                        }
                    }
                    impl<'de> _serde::Deserialize<'de> for __Field {
                        #[inline]
                        fn deserialize<__D>(
                            __deserializer: __D,
                        ) -> _serde::__private::Result<Self, __D::Error>
                        where
                            __D: _serde::Deserializer<'de>,
                        {
                            _serde::Deserializer::deserialize_identifier(
                                __deserializer,
                                __FieldVisitor,
                            )
                        }
                    }
                    struct __Visitor<'de> {
                        marker: _serde::__private::PhantomData<StarRelationship>,
                        lifetime: _serde::__private::PhantomData<&'de ()>,
                    }
                    impl<'de> _serde::de::Visitor<'de> for __Visitor<'de> {
                        type Value = StarRelationship;
                        fn expecting(
                            &self,
                            __formatter: &mut _serde::__private::Formatter,
                        ) -> _serde::__private::fmt::Result {
                            _serde::__private::Formatter::write_str(
                                __formatter,
                                "struct StarRelationship",
                            )
                        }
                        #[inline]
                        fn visit_seq<__A>(
                            self,
                            mut __seq: __A,
                        ) -> _serde::__private::Result<Self::Value, __A::Error>
                        where
                            __A: _serde::de::SeqAccess<'de>,
                        {
                            let __field0 = match match _serde::de::SeqAccess::next_element::<
                                OperationId,
                            >(&mut __seq)
                            {
                                _serde::__private::Ok(__val) => __val,
                                _serde::__private::Err(__err) => {
                                    return _serde::__private::Err(__err);
                                }
                            } {
                                _serde::__private::Some(__value) => __value,
                                _serde::__private::None => {
                                    return _serde::__private::Err(
                                        _serde::de::Error::invalid_length(
                                            0usize,
                                            &"struct StarRelationship with 4 elements",
                                        ),
                                    );
                                }
                            };
                            let __field1 = match match _serde::de::SeqAccess::next_element::<
                                Option<usize>,
                            >(&mut __seq)
                            {
                                _serde::__private::Ok(__val) => __val,
                                _serde::__private::Err(__err) => {
                                    return _serde::__private::Err(__err);
                                }
                            } {
                                _serde::__private::Some(__value) => __value,
                                _serde::__private::None => {
                                    return _serde::__private::Err(
                                        _serde::de::Error::invalid_length(
                                            1usize,
                                            &"struct StarRelationship with 4 elements",
                                        ),
                                    );
                                }
                            };
                            let __field2 = match match _serde::de::SeqAccess::next_element::<
                                Vec<usize>,
                            >(&mut __seq)
                            {
                                _serde::__private::Ok(__val) => __val,
                                _serde::__private::Err(__err) => {
                                    return _serde::__private::Err(__err);
                                }
                            } {
                                _serde::__private::Some(__value) => __value,
                                _serde::__private::None => {
                                    return _serde::__private::Err(
                                        _serde::de::Error::invalid_length(
                                            2usize,
                                            &"struct StarRelationship with 4 elements",
                                        ),
                                    );
                                }
                            };
                            let __field3 = match match _serde::de::SeqAccess::next_element::<
                                Vec<usize>,
                            >(&mut __seq)
                            {
                                _serde::__private::Ok(__val) => __val,
                                _serde::__private::Err(__err) => {
                                    return _serde::__private::Err(__err);
                                }
                            } {
                                _serde::__private::Some(__value) => __value,
                                _serde::__private::None => {
                                    return _serde::__private::Err(
                                        _serde::de::Error::invalid_length(
                                            3usize,
                                            &"struct StarRelationship with 4 elements",
                                        ),
                                    );
                                }
                            };
                            _serde::__private::Ok(StarRelationship {
                                operation_id: __field0,
                                step: __field1,
                                input_star_ids: __field2,
                                output_star_ids: __field3,
                            })
                        }
                        #[inline]
                        fn visit_map<__A>(
                            self,
                            mut __map: __A,
                        ) -> _serde::__private::Result<Self::Value, __A::Error>
                        where
                            __A: _serde::de::MapAccess<'de>,
                        {
                            let mut __field0: _serde::__private::Option<OperationId> =
                                _serde::__private::None;
                            let mut __field1: _serde::__private::Option<Option<usize>> =
                                _serde::__private::None;
                            let mut __field2: _serde::__private::Option<Vec<usize>> =
                                _serde::__private::None;
                            let mut __field3: _serde::__private::Option<Vec<usize>> =
                                _serde::__private::None;
                            while let _serde::__private::Some(__key) =
                                match _serde::de::MapAccess::next_key::<__Field>(&mut __map) {
                                    _serde::__private::Ok(__val) => __val,
                                    _serde::__private::Err(__err) => {
                                        return _serde::__private::Err(__err);
                                    }
                                }
                            {
                                match __key {
                                    __Field::__field0 => {
                                        if _serde::__private::Option::is_some(&__field0) {
                                            return _serde::__private::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field(
                                                    "operation_id",
                                                ),
                                            );
                                        }
                                        __field0 = _serde::__private::Some(
                                            match _serde::de::MapAccess::next_value::<OperationId>(
                                                &mut __map,
                                            ) {
                                                _serde::__private::Ok(__val) => __val,
                                                _serde::__private::Err(__err) => {
                                                    return _serde::__private::Err(__err);
                                                }
                                            },
                                        );
                                    }
                                    __Field::__field1 => {
                                        if _serde::__private::Option::is_some(&__field1) {
                                            return _serde::__private::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field(
                                                    "step",
                                                ),
                                            );
                                        }
                                        __field1 = _serde::__private::Some(
                                            match _serde::de::MapAccess::next_value::<Option<usize>>(
                                                &mut __map,
                                            ) {
                                                _serde::__private::Ok(__val) => __val,
                                                _serde::__private::Err(__err) => {
                                                    return _serde::__private::Err(__err);
                                                }
                                            },
                                        );
                                    }
                                    __Field::__field2 => {
                                        if _serde::__private::Option::is_some(&__field2) {
                                            return _serde::__private::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field(
                                                    "input_star_ids",
                                                ),
                                            );
                                        }
                                        __field2 = _serde::__private::Some(
                                            match _serde::de::MapAccess::next_value::<Vec<usize>>(
                                                &mut __map,
                                            ) {
                                                _serde::__private::Ok(__val) => __val,
                                                _serde::__private::Err(__err) => {
                                                    return _serde::__private::Err(__err);
                                                }
                                            },
                                        );
                                    }
                                    __Field::__field3 => {
                                        if _serde::__private::Option::is_some(&__field3) {
                                            return _serde::__private::Err(
                                                <__A::Error as _serde::de::Error>::duplicate_field(
                                                    "output_star_ids",
                                                ),
                                            );
                                        }
                                        __field3 = _serde::__private::Some(
                                            match _serde::de::MapAccess::next_value::<Vec<usize>>(
                                                &mut __map,
                                            ) {
                                                _serde::__private::Ok(__val) => __val,
                                                _serde::__private::Err(__err) => {
                                                    return _serde::__private::Err(__err);
                                                }
                                            },
                                        );
                                    }
                                    _ => {
                                        let _ = match _serde::de::MapAccess::next_value::<
                                            _serde::de::IgnoredAny,
                                        >(
                                            &mut __map
                                        ) {
                                            _serde::__private::Ok(__val) => __val,
                                            _serde::__private::Err(__err) => {
                                                return _serde::__private::Err(__err);
                                            }
                                        };
                                    }
                                }
                            }
                            let __field0 = match __field0 {
                                _serde::__private::Some(__field0) => __field0,
                                _serde::__private::None => {
                                    match _serde::__private::de::missing_field("operation_id") {
                                        _serde::__private::Ok(__val) => __val,
                                        _serde::__private::Err(__err) => {
                                            return _serde::__private::Err(__err);
                                        }
                                    }
                                }
                            };
                            let __field1 = match __field1 {
                                _serde::__private::Some(__field1) => __field1,
                                _serde::__private::None => {
                                    match _serde::__private::de::missing_field("step") {
                                        _serde::__private::Ok(__val) => __val,
                                        _serde::__private::Err(__err) => {
                                            return _serde::__private::Err(__err);
                                        }
                                    }
                                }
                            };
                            let __field2 = match __field2 {
                                _serde::__private::Some(__field2) => __field2,
                                _serde::__private::None => {
                                    match _serde::__private::de::missing_field("input_star_ids") {
                                        _serde::__private::Ok(__val) => __val,
                                        _serde::__private::Err(__err) => {
                                            return _serde::__private::Err(__err);
                                        }
                                    }
                                }
                            };
                            let __field3 = match __field3 {
                                _serde::__private::Some(__field3) => __field3,
                                _serde::__private::None => {
                                    match _serde::__private::de::missing_field("output_star_ids") {
                                        _serde::__private::Ok(__val) => __val,
                                        _serde::__private::Err(__err) => {
                                            return _serde::__private::Err(__err);
                                        }
                                    }
                                }
                            };
                            _serde::__private::Ok(StarRelationship {
                                operation_id: __field0,
                                step: __field1,
                                input_star_ids: __field2,
                                output_star_ids: __field3,
                            })
                        }
                    }
                    const FIELDS: &'static [&'static str] =
                        &["operation_id", "step", "input_star_ids", "output_star_ids"];
                    _serde::Deserializer::deserialize_struct(
                        __deserializer,
                        "StarRelationship",
                        FIELDS,
                        __Visitor {
                            marker: _serde::__private::PhantomData::<StarRelationship>,
                            lifetime: _serde::__private::PhantomData,
                        },
                    )
                }
            }
        };
        /// We assume there's a root star that is the ordered concatenation of the DNN's input variables.
        /// This facilitates the representation of each star.
        pub trait StarSet<D: 'static + Dimension> {
            /// Get the Graph from the DNN
            fn get_graph(&self) -> &Graph;
            /// Get DNN
            fn get_dnn(&self) -> &DNN;
            /// Get the id of the root star in the starset.
            fn get_root_id(&self) -> StarId;
            /// Get the DNN/Graph `RepresentationId` that a star corresponds to
            fn get_star_representation_id(&self, star_id: StarId) -> RepresentationId;
            /// Get a reference to a star
            fn get_star(&self, star_id: StarId) -> Ref<Star<D>>;
            /// Gets a relationship
            fn get_relationship(
                &self,
                relationship_id: StarRelationshipId,
            ) -> Ref<StarRelationship>;
            /// Add a star
            /// Requires interior mutability
            fn add_star(
                &self,
                star: Star<D>,
                representation_id: RepresentationId,
                axis_aligned_input_bounds: Bounds<D>,
            ) -> StarId;
            /// Adds a relationship
            /// Requires interior mutability
            fn add_relationship(&self, star_rel: StarRelationship) -> StarRelationshipId;
        }
        pub trait StarSet2: StarSet<Ix2> {
            # ! [doc = " # TODO: discuss the API of this trait"] # ! [doc = " # TODO: discuss assumptions of this trait"]            /// Get the dimension of the DNN input
            fn get_input_dim(&self) -> usize;
            /// TODO: Implement with a cache because it is expensive
            fn get_axis_aligned_input_bounds(&self, star_id: StarId) -> &Bounds1;
            /// Expand an operation from its inputs to produce the children and adds them to the `StarSet`.
            ///
            /// # Description
            ///
            /// Each non-empty child star is stored as a separate `StarNode` in the `StarSet`.
            ///
            /// # Invariants
            ///
            /// The stars pointed to by `input_stars_ids` must be those that correspond to the `RepresentationId`s of the inputs to the operation and they must be in the same order.
            ///
            /// # Arguments
            /// * `operation_id` - The operation of the DNN on which to expand the star set.
            /// * `input_star_ids` - The ordered ids of the `star`s that are used as inputs to the operation.
            fn expand(
                &self,
                operation_id: OperationId,
                input_star_ids: Vec<StarId>,
            ) -> StarRelationshipId {
                if !!input_star_ids.is_empty() {
                    ::core::panicking::panic("assertion failed: !input_star_ids.is_empty()")
                };
                let operation_node = self.get_graph().get_operation_node(&operation_id).unwrap();
                let repr_ids = input_star_ids
                    .iter()
                    .map(|star_id| self.get_star_representation_id(*star_id))
                    .collect::<Vec<_>>();
                match (&repr_ids.len(), &operation_node.get_input_ids().len()) {
                    (left_val, right_val) => {
                        if !(*left_val == *right_val) {
                            let kind = ::core::panicking::AssertKind::Eq;
                            ::core::panicking::assert_failed(
                                kind,
                                &*left_val,
                                &*right_val,
                                ::core::option::Option::None,
                            );
                        }
                    }
                };
                let step_opt = repr_ids.first().unwrap().operation_step;
                if !!repr_ids
                    .into_iter()
                    .any(|repr_id| repr_id.operation_step != step_opt)
                {
                    :: core :: panicking :: panic ("assertion failed: !repr_ids.into_iter().any(|repr_id| repr_id.operation_step != step_opt)")
                };
                let next_step_opt = match (operation_node.get_operation().num_steps(), step_opt) {
                    (None | Some(1), None) => None,
                    (Some(num_steps), Some(step)) if step + 2 == num_steps => None,
                    (Some(num_steps), Some(step)) if step + 2 < num_steps => Some(step + 1),
                    (Some(_), None) => Some(0),
                    _ => ::core::panicking::panic("explicit panic"),
                };
                let parent_bounds = input_star_ids
                    .iter()
                    .map(|&star_id| self.get_axis_aligned_input_bounds(star_id))
                    .collect::<Vec<_>>();
                let stars = input_star_ids
                    .iter()
                    .map(|&node_id| self.get_star(node_id))
                    .collect::<Vec<_>>();
                let (child_stars, child_input_bounds, _same_output_bounds) = operation_node
                    .get_operation()
                    .downcast_ref::<StarOperation>()
                    .forward_star(stars, next_step_opt, parent_bounds);
                let child_star_ids = child_stars
                    .into_iter()
                    .zip(operation_node.get_output_ids().clone().into_iter())
                    .zip(child_input_bounds.into_iter())
                    .map(|((star, repr_id), child_input_bounds)| {
                        self.add_star(star, repr_id, child_input_bounds)
                    })
                    .collect();
                let star_rel = StarRelationship {
                    operation_id,
                    step: next_step_opt,
                    input_star_ids,
                    output_star_ids: child_star_ids,
                };
                self.add_relationship(star_rel)
            }
        }
    }
}
pub mod tensorshape {
    #![allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
    use serde::{Deserialize, Serialize};
    use std::fmt::Display;
    use std::ops::Index;
    pub struct TensorShape {
        dims: Vec<Option<usize>>,
    }
    #[automatically_derived]
    #[allow(unused_qualifications)]
    impl ::core::fmt::Debug for TensorShape {
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            match *self {
                TensorShape {
                    dims: ref __self_0_0,
                } => {
                    let debug_trait_builder =
                        &mut ::core::fmt::Formatter::debug_struct(f, "TensorShape");
                    let _ = ::core::fmt::DebugStruct::field(
                        debug_trait_builder,
                        "dims",
                        &&(*__self_0_0),
                    );
                    ::core::fmt::DebugStruct::finish(debug_trait_builder)
                }
            }
        }
    }
    #[automatically_derived]
    #[allow(unused_qualifications)]
    impl ::core::clone::Clone for TensorShape {
        #[inline]
        fn clone(&self) -> TensorShape {
            match *self {
                TensorShape {
                    dims: ref __self_0_0,
                } => TensorShape {
                    dims: ::core::clone::Clone::clone(&(*__self_0_0)),
                },
            }
        }
    }
    #[doc(hidden)]
    #[allow(non_upper_case_globals, unused_attributes, unused_qualifications)]
    const _: () = {
        #[allow(unused_extern_crates, clippy::useless_attribute)]
        extern crate serde as _serde;
        #[automatically_derived]
        impl _serde::Serialize for TensorShape {
            fn serialize<__S>(
                &self,
                __serializer: __S,
            ) -> _serde::__private::Result<__S::Ok, __S::Error>
            where
                __S: _serde::Serializer,
            {
                let mut __serde_state = match _serde::Serializer::serialize_struct(
                    __serializer,
                    "TensorShape",
                    false as usize + 1,
                ) {
                    _serde::__private::Ok(__val) => __val,
                    _serde::__private::Err(__err) => {
                        return _serde::__private::Err(__err);
                    }
                };
                match _serde::ser::SerializeStruct::serialize_field(
                    &mut __serde_state,
                    "dims",
                    &self.dims,
                ) {
                    _serde::__private::Ok(__val) => __val,
                    _serde::__private::Err(__err) => {
                        return _serde::__private::Err(__err);
                    }
                };
                _serde::ser::SerializeStruct::end(__serde_state)
            }
        }
    };
    #[doc(hidden)]
    #[allow(non_upper_case_globals, unused_attributes, unused_qualifications)]
    const _: () = {
        #[allow(unused_extern_crates, clippy::useless_attribute)]
        extern crate serde as _serde;
        #[automatically_derived]
        impl<'de> _serde::Deserialize<'de> for TensorShape {
            fn deserialize<__D>(__deserializer: __D) -> _serde::__private::Result<Self, __D::Error>
            where
                __D: _serde::Deserializer<'de>,
            {
                #[allow(non_camel_case_types)]
                enum __Field {
                    __field0,
                    __ignore,
                }
                struct __FieldVisitor;
                impl<'de> _serde::de::Visitor<'de> for __FieldVisitor {
                    type Value = __Field;
                    fn expecting(
                        &self,
                        __formatter: &mut _serde::__private::Formatter,
                    ) -> _serde::__private::fmt::Result {
                        _serde::__private::Formatter::write_str(__formatter, "field identifier")
                    }
                    fn visit_u64<__E>(
                        self,
                        __value: u64,
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            0u64 => _serde::__private::Ok(__Field::__field0),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                    fn visit_str<__E>(
                        self,
                        __value: &str,
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            "dims" => _serde::__private::Ok(__Field::__field0),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                    fn visit_bytes<__E>(
                        self,
                        __value: &[u8],
                    ) -> _serde::__private::Result<Self::Value, __E>
                    where
                        __E: _serde::de::Error,
                    {
                        match __value {
                            b"dims" => _serde::__private::Ok(__Field::__field0),
                            _ => _serde::__private::Ok(__Field::__ignore),
                        }
                    }
                }
                impl<'de> _serde::Deserialize<'de> for __Field {
                    #[inline]
                    fn deserialize<__D>(
                        __deserializer: __D,
                    ) -> _serde::__private::Result<Self, __D::Error>
                    where
                        __D: _serde::Deserializer<'de>,
                    {
                        _serde::Deserializer::deserialize_identifier(__deserializer, __FieldVisitor)
                    }
                }
                struct __Visitor<'de> {
                    marker: _serde::__private::PhantomData<TensorShape>,
                    lifetime: _serde::__private::PhantomData<&'de ()>,
                }
                impl<'de> _serde::de::Visitor<'de> for __Visitor<'de> {
                    type Value = TensorShape;
                    fn expecting(
                        &self,
                        __formatter: &mut _serde::__private::Formatter,
                    ) -> _serde::__private::fmt::Result {
                        _serde::__private::Formatter::write_str(__formatter, "struct TensorShape")
                    }
                    #[inline]
                    fn visit_seq<__A>(
                        self,
                        mut __seq: __A,
                    ) -> _serde::__private::Result<Self::Value, __A::Error>
                    where
                        __A: _serde::de::SeqAccess<'de>,
                    {
                        let __field0 = match match _serde::de::SeqAccess::next_element::<
                            Vec<Option<usize>>,
                        >(&mut __seq)
                        {
                            _serde::__private::Ok(__val) => __val,
                            _serde::__private::Err(__err) => {
                                return _serde::__private::Err(__err);
                            }
                        } {
                            _serde::__private::Some(__value) => __value,
                            _serde::__private::None => {
                                return _serde::__private::Err(_serde::de::Error::invalid_length(
                                    0usize,
                                    &"struct TensorShape with 1 element",
                                ));
                            }
                        };
                        _serde::__private::Ok(TensorShape { dims: __field0 })
                    }
                    #[inline]
                    fn visit_map<__A>(
                        self,
                        mut __map: __A,
                    ) -> _serde::__private::Result<Self::Value, __A::Error>
                    where
                        __A: _serde::de::MapAccess<'de>,
                    {
                        let mut __field0: _serde::__private::Option<Vec<Option<usize>>> =
                            _serde::__private::None;
                        while let _serde::__private::Some(__key) =
                            match _serde::de::MapAccess::next_key::<__Field>(&mut __map) {
                                _serde::__private::Ok(__val) => __val,
                                _serde::__private::Err(__err) => {
                                    return _serde::__private::Err(__err);
                                }
                            }
                        {
                            match __key {
                                __Field::__field0 => {
                                    if _serde::__private::Option::is_some(&__field0) {
                                        return _serde::__private::Err(
                                            <__A::Error as _serde::de::Error>::duplicate_field(
                                                "dims",
                                            ),
                                        );
                                    }
                                    __field0 = _serde::__private::Some(
                                        match _serde::de::MapAccess::next_value::<Vec<Option<usize>>>(
                                            &mut __map,
                                        ) {
                                            _serde::__private::Ok(__val) => __val,
                                            _serde::__private::Err(__err) => {
                                                return _serde::__private::Err(__err);
                                            }
                                        },
                                    );
                                }
                                _ => {
                                    let _ = match _serde::de::MapAccess::next_value::<
                                        _serde::de::IgnoredAny,
                                    >(&mut __map)
                                    {
                                        _serde::__private::Ok(__val) => __val,
                                        _serde::__private::Err(__err) => {
                                            return _serde::__private::Err(__err);
                                        }
                                    };
                                }
                            }
                        }
                        let __field0 = match __field0 {
                            _serde::__private::Some(__field0) => __field0,
                            _serde::__private::None => {
                                match _serde::__private::de::missing_field("dims") {
                                    _serde::__private::Ok(__val) => __val,
                                    _serde::__private::Err(__err) => {
                                        return _serde::__private::Err(__err);
                                    }
                                }
                            }
                        };
                        _serde::__private::Ok(TensorShape { dims: __field0 })
                    }
                }
                const FIELDS: &'static [&'static str] = &["dims"];
                _serde::Deserializer::deserialize_struct(
                    __deserializer,
                    "TensorShape",
                    FIELDS,
                    __Visitor {
                        marker: _serde::__private::PhantomData::<TensorShape>,
                        lifetime: _serde::__private::PhantomData,
                    },
                )
            }
        }
    };
    impl TensorShape {
        /// # Panics
        pub fn new(dims: Vec<Option<usize>>) -> Self {
            if !dims.iter().any(Option::is_some) {
                ::core::panicking::panic("assertion failed: dims.iter().any(Option::is_some)")
            };
            Self { dims }
        }
        pub fn rank(&self) -> usize {
            self.dims.len()
        }
        pub fn dims(&self) -> usize {
            self.dims.iter().fold(1, |a, b| a * b.unwrap_or(1))
        }
        pub fn is_fully_defined(&self) -> bool {
            self.dims.iter().all(Option::is_some)
        }
        pub fn as_defined_slice(&self) -> Option<Vec<usize>> {
            let slice: Vec<usize> = self.dims.iter().filter_map(|&x| x).collect();
            if slice.len() == self.rank() {
                Some(slice)
            } else {
                None
            }
        }
        pub fn is_compatible_with(&self, other: &Self) -> bool {
            if self.dims == <[_]>::into_vec(box [None]) {
                return true;
            }
            if self.dims.len() != other.dims.len() {
                return false;
            }
            self.dims
                .iter()
                .zip(other.dims.iter())
                .all(|(x, y)| match (x, y) {
                    (Some(a), Some(b)) => a == b,
                    _ => true,
                })
        }
    }
    impl From<TensorShape> for Vec<Option<usize>> {
        fn from(ts: TensorShape) -> Self {
            ts.dims
        }
    }
    impl Index<isize> for TensorShape {
        type Output = Option<usize>;
        fn index(&self, mut idx: isize) -> &Option<usize> {
            if idx < 0 {
                idx += self.dims.len() as isize;
            }
            if true {
                if !(idx >= 0) {
                    ::core::panicking::panic_fmt(::core::fmt::Arguments::new_v1(
                        &["idx ", " < 0"],
                        &[::core::fmt::ArgumentV1::new_display(&idx)],
                    ))
                };
            };
            &self.dims[idx as usize]
        }
    }
    impl From<Vec<usize>> for TensorShape {
        fn from(v: Vec<usize>) -> Self {
            Self {
                dims: v.into_iter().map(Some).collect(),
            }
        }
    }
    impl Display for TensorShape {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::result::Result<(), std::fmt::Error> {
            let strs = self
                .dims
                .iter()
                .map(|opt| opt.map_or("None".to_owned(), |x| x.to_string()))
                .collect::<Vec<String>>();
            f.write_fmt(::core::fmt::Arguments::new_v1(
                &["(", ")"],
                &[::core::fmt::ArgumentV1::new_display(&strs.join(", "))],
            ))
        }
    }
}
pub mod util {
    //! Utility functions
    #![allow(non_snake_case)]
    use itertools::iproduct;
    use ndarray::{s, Axis, Slice};
    use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
    use ndarray_linalg::Scalar;
    use ndarray_linalg::SVD;
    use ndarray_stats::QuantileExt;
    use num::Float;
    use rand::Rng;
    use std::cmp::max;
    use std::cmp::Ordering;
    use std::fmt::Debug;
    use std::iter::Sum;
    pub fn gaussian_logp(x: &ArrayView1<f64>, mu: &ArrayView1<f64>, std: &ArrayView1<f64>) -> f64 {
        let pre_sum: Array1<f64> = ((&(x - mu) / &(std + f64::epsilon())).mapv(f64::square)
            + std.mapv(f64::ln) * (2.)
            + std::f64::consts::TAU.ln())
            * (-0.5);
        pre_sum.sum()
    }
    pub fn diag_gaussian_accept_reject<R: Rng>(
        x: &ArrayView1<f64>,
        mu: &ArrayView1<f64>,
        sigma: &ArrayView1<f64>,
        rng: &mut R,
    ) -> bool {
        let likelihood = gaussian_logp(x, mu, sigma).exp();
        let sample: f64 = rng.gen();
        sample < likelihood
    }
    /// # Panics
    pub fn matrix_cond(A: &Array2<f64>, A_inv: &Array2<f64>) -> f64 {
        let (_, sigma, _) = A.svd(false, false).unwrap();
        let (_, inv_sigma, _) = A_inv.svd(false, false).unwrap();
        return sigma.max_skipnan() * inv_sigma.max_skipnan();
    }
    pub fn l2_norm(x: ArrayView1<f64>) -> f64 {
        x.dot(&x).sqrt()
    }
    /// # Panics
    pub fn pinv(x: &Array2<f64>) -> Array2<f64> {
        let (u_opt, sigma, vt_opt) = x.svd(true, true).unwrap();
        let u = u_opt.unwrap();
        let vt = vt_opt.unwrap();
        let sig_diag = &sigma.map(|x| if *x < 1e-10 { 0. } else { 1. / x });
        let mut sig_base = Array2::eye(max(u.nrows(), vt.nrows()));
        sig_base
            .diag_mut()
            .slice_mut(match ..sig_diag.len() {
                r => {
                    let in_dim = ::ndarray::SliceNextDim::next_in_dim(
                        &r,
                        ::std::marker::PhantomData::<::ndarray::Ix0>,
                    );
                    let out_dim = ::ndarray::SliceNextDim::next_out_dim(
                        &r,
                        ::std::marker::PhantomData::<::ndarray::Ix0>,
                    );
                    #[allow(unsafe_code)]
                    unsafe {
                        ::ndarray::SliceInfo::new_unchecked(
                            [<::ndarray::SliceInfoElem as ::std::convert::From<_>>::from(
                                r,
                            )],
                            in_dim,
                            out_dim,
                        )
                    }
                }
            })
            .assign(sig_diag);
        let sig = sig_base
            .slice_axis(Axis(0), Slice::from(..vt.nrows()))
            .to_owned();
        let final_sig = sig.slice_axis(Axis(1), Slice::from(..u.nrows()));
        vt.t().dot(&final_sig.dot(&u.t()))
    }
    /// # Panics
    pub fn ensure_spd(A: &Array2<f64>) -> Array2<f64> {
        let B = (A + &A.t()) / 2.;
        let (_, sigma, vt_opt) = A.svd(false, true).unwrap();
        let vt = vt_opt.unwrap();
        let H = vt.t().dot(&sigma).dot(&vt);
        let mut a_hat = (B + H) / 2.;
        a_hat = (&a_hat + &a_hat.t()) / 2.;
        a_hat
    }
    pub fn embed_identity(A: &Array2<f64>, dim_opt: Option<usize>) -> Array2<f64> {
        let dim = match dim_opt {
            Some(dim) => dim,
            None => max(A.nrows(), A.ncols()),
        };
        let mut eye = Array2::eye(dim);
        eye.slice_mut(match ..A.nrows() {
            r => match ..A.ncols() {
                r => {
                    let in_dim = ::ndarray::SliceNextDim::next_in_dim(
                        &r,
                        ::ndarray::SliceNextDim::next_in_dim(
                            &r,
                            ::std::marker::PhantomData::<::ndarray::Ix0>,
                        ),
                    );
                    let out_dim = ::ndarray::SliceNextDim::next_out_dim(
                        &r,
                        ::ndarray::SliceNextDim::next_out_dim(
                            &r,
                            ::std::marker::PhantomData::<::ndarray::Ix0>,
                        ),
                    );
                    #[allow(unsafe_code)]
                    unsafe {
                        ::ndarray::SliceInfo::new_unchecked(
                            [
                                <::ndarray::SliceInfoElem as ::std::convert::From<_>>::from(r),
                                <::ndarray::SliceInfoElem as ::std::convert::From<_>>::from(r),
                            ],
                            in_dim,
                            out_dim,
                        )
                    }
                }
            },
        })
        .assign(A);
        eye
    }
    /// Returns a 2D array of D\[i,j\] = AB\[i,j\] if A\[i,j\] >= 0 and D\[i,j\] = AC\[i,j\] if A\[i,j\] < 0.
    ///
    /// * `A` - The base array of shape `mn`
    /// * `B` - The positive array of shape `nk`
    /// * `C` - The negative array of shape `nk`
    ///
    /// # Panics
    pub fn signed_matmul<T: Float + Sum + Debug>(
        A: &ArrayView2<T>,
        B: &ArrayView2<T>,
        C: &ArrayView2<T>,
    ) -> Array2<T> {
        if true {
            match (&A.ncols(), &B.nrows()) {
                (left_val, right_val) => {
                    if !(*left_val == *right_val) {
                        let kind = ::core::panicking::AssertKind::Eq;
                        ::core::panicking::assert_failed(
                            kind,
                            &*left_val,
                            &*right_val,
                            ::core::option::Option::None,
                        );
                    }
                }
            };
        };
        if true {
            match (&A.ncols(), &C.nrows()) {
                (left_val, right_val) => {
                    if !(*left_val == *right_val) {
                        let kind = ::core::panicking::AssertKind::Eq;
                        ::core::panicking::assert_failed(
                            kind,
                            &*left_val,
                            &*right_val,
                            ::core::option::Option::None,
                        );
                    }
                }
            };
        };
        let mut out = Array2::zeros([A.nrows(), B.ncols()]);
        ::itertools::Itertools::cartesian_product(
            ::itertools::__std_iter::IntoIterator::into_iter(0..A.nrows()),
            ::itertools::__std_iter::IntoIterator::into_iter(0..B.ncols()),
        )
        .for_each(|(i, j)| {
            out[[i, j]] = (0..A.ncols())
                .map(|k| {
                    if A[[i, k]] >= T::zero() {
                        A[[i, k]] * B[[k, j]]
                    } else {
                        A[[i, k]] * C[[k, j]]
                    }
                })
                .sum();
        });
        out
    }
    /// # Panics
    pub fn signed_dot<T: Float + Sum + Debug>(
        A: &ArrayView2<T>,
        B: &ArrayView1<T>,
        C: &ArrayView1<T>,
    ) -> Array1<T> {
        if true {
            match (&A.ncols(), &B.len()) {
                (left_val, right_val) => {
                    if !(*left_val == *right_val) {
                        let kind = ::core::panicking::AssertKind::Eq;
                        ::core::panicking::assert_failed(
                            kind,
                            &*left_val,
                            &*right_val,
                            ::core::option::Option::None,
                        );
                    }
                }
            };
        };
        if true {
            match (&A.ncols(), &C.len()) {
                (left_val, right_val) => {
                    if !(*left_val == *right_val) {
                        let kind = ::core::panicking::AssertKind::Eq;
                        ::core::panicking::assert_failed(
                            kind,
                            &*left_val,
                            &*right_val,
                            ::core::option::Option::None,
                        );
                    }
                }
            };
        };
        let mut out = Array1::zeros(A.nrows());
        (0..A.nrows()).for_each(|i| {
            out[[i]] = (0..A.ncols())
                .map(|k| {
                    if A[[i, k]] >= T::zero() {
                        A[[i, k]] * B[[k]]
                    } else {
                        A[[i, k]] * C[[k]]
                    }
                })
                .sum();
        });
        out
    }
    pub trait ArenaLike<T> {
        fn push_node(&mut self, data: T) -> usize;
    }
    impl<T> ArenaLike<T> for Vec<T> {
        fn push_node(&mut self, data: T) -> usize {
            let new_id = self.len();
            self.push(data);
            new_id
        }
    }
    pub struct FstOrdTuple<A: Ord, B>(pub (A, B));
    impl<A: Ord, B> ::core::marker::StructuralEq for FstOrdTuple<A, B> {}
    #[automatically_derived]
    #[allow(unused_qualifications)]
    impl<A: ::core::cmp::Eq + Ord, B: ::core::cmp::Eq> ::core::cmp::Eq for FstOrdTuple<A, B> {
        #[inline]
        #[doc(hidden)]
        #[no_coverage]
        fn assert_receiver_is_total_eq(&self) -> () {
            {
                let _: ::core::cmp::AssertParamIsEq<(A, B)>;
            }
        }
    }
    impl<A: Ord, B> ::core::marker::StructuralPartialEq for FstOrdTuple<A, B> {}
    #[automatically_derived]
    #[allow(unused_qualifications)]
    impl<A: ::core::cmp::PartialEq + Ord, B: ::core::cmp::PartialEq> ::core::cmp::PartialEq
        for FstOrdTuple<A, B>
    {
        #[inline]
        fn eq(&self, other: &FstOrdTuple<A, B>) -> bool {
            match *other {
                FstOrdTuple(ref __self_1_0) => match *self {
                    FstOrdTuple(ref __self_0_0) => (*__self_0_0) == (*__self_1_0),
                },
            }
        }
        #[inline]
        fn ne(&self, other: &FstOrdTuple<A, B>) -> bool {
            match *other {
                FstOrdTuple(ref __self_1_0) => match *self {
                    FstOrdTuple(ref __self_0_0) => (*__self_0_0) != (*__self_1_0),
                },
            }
        }
    }
    impl<A: Ord, B: PartialEq> PartialOrd for FstOrdTuple<A, B> {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            self.0 .0.partial_cmp(&other.0 .0)
        }
    }
    impl<A: Ord, B: Eq> Ord for FstOrdTuple<A, B> {
        fn cmp(&self, other: &Self) -> Ordering {
            self.0 .0.cmp(&other.0 .0)
        }
    }
}
pub type NNVFloat = f64;
pub mod trunks {
    use crate::polytope::Polytope;
    use ndarray::{Array1, Array2, Axis};
    pub fn halfspace_gaussian_cdf(
        coeffs: Array1<f64>,
        rhs: f64,
        mu: &Array1<f64>,
        sigma: &Array1<f64>,
    ) -> f64 {
        let mut rng = rand::thread_rng();
        let polytope = Polytope::new(
            coeffs.insert_axis(Axis(0)),
            Array1::from_vec(<[_]>::into_vec(box [rhs])),
        );
        let mut truncnorm = polytope.get_truncnorm_distribution(
            mu.view(),
            Array2::from_diag(sigma).view(),
            3,
            1e-10,
        );
        truncnorm.cdf(1000, &mut rng)
    }
}
