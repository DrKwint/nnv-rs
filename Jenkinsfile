pipeline {
    agent any

    stages {
        stage('Build') {
            steps {
                sh 'cargo build --features lp_coincbc'
                sh 'cargo build --features lp_gurobi'
            }
        }

        stage('Test') {
            steps {
                sh 'cargo test --features lp_coincbc'
                sh 'cargo test --features lp_gurobi'
            }
        }

        stage('Bench') {
            steps {
                sh 'RUST_LOG=trace cargo bench --features lp_coincbc'
                sh 'RUST_LOG=trace cargo bench --features lp_gurobi'
                sh 'cargo bench --features lp_coincbc'
                sh 'cargo bench --features lp_gurobi'
            }
        }
    }
}
