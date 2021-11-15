pipeline {
    agent any

    stages {
        stage('Build') {
            environment {
                CARGO_HOME = '/usr/local/cargo/'
            }

            steps {
                sh '$CARGO_HOME/bin/cargo build'
            }
        }
    }
}
