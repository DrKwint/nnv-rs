use ndarray::Array2;
use ndarray::Axis;
use ndarray::Slice;
use ndarray_linalg::EigVals;
use ndarray_linalg::SVD;

pub fn pinv(x: &Array2<f64>) -> Array2<f64> {
    let (u_opt, sigma, vt_opt) = x.svd(true, true).unwrap();
    let u = u_opt.unwrap();
    let vt = vt_opt.unwrap();
    let sig_square = Array2::from_diag(&sigma.map(|x| if *x < 1e-10 { 0. } else { 1. / x }));
    let sig_base = Array2::eye(vt.nrows());
    let sig = sig_base.slice_axis(Axis(1), Slice::from(..sig_square.nrows()));
    vt.t().dot(&sig.dot(&u.t()))
}

pub fn ensure_spd(A: Array2<f64>) -> Array2<f64> {
    let B = (&A + &A.t()) / 2.;
    let (_, sigma, vt_opt) = A.svd(false, true).unwrap();
    let vt = vt_opt.unwrap();
    let H = vt.t().dot(&sigma).dot(&vt);
    let mut a_hat = (B + H) / 2.;
    // ensure symmetry
    a_hat = (&a_hat + &a_hat.t()) / 2.;
    let min_eig = a_hat.eigvals().unwrap();
    println!("min_eig {}", min_eig);
    a_hat
}
