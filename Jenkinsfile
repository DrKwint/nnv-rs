pipeline {
    agent any

    stages {
        stage('Build') {
            steps {
                sh 'cargo build --features lp_coincbc,blas_openblas-system'
                sh 'cargo build --features lp_gurobi,blas_intel-mkl'
            }
        }

        stage('Test') {
            steps {
                sh 'cargo test --features lp_gurobi,blas_intel-mkl'
            }
        }

        stage('Bench') {
            steps {
                sh 'RUST_LOG=trace cargo bench --features lp_coincbc,blas_intel-mkl'
                sh 'RUST_LOG=trace cargo bench --features lp_gurobi,blas_intel-mkl'
                sh 'cargo bench --features lp_coincbc,blas_intel-mkl'
                sh 'cargo bench --features lp_gurobi,blas_intel-mkl'
            }
        }
    }
}
