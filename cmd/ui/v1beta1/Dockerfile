# --- Clone the kubeflow/kubeflow code ---
FROM ubuntu AS fetch-kubeflow-kubeflow

RUN apt-get update && apt-get install git -y

WORKDIR /kf
COPY ./pkg/ui/v1beta1/frontend/COMMIT ./
RUN git clone https://github.com/kubeflow/kubeflow.git && \
    COMMIT=$(cat ./COMMIT) && \
    cd kubeflow && \
    git checkout $COMMIT

# --- Build the frontend kubeflow library ---
FROM node:16.20.2-bullseye AS frontend-kubeflow-lib

WORKDIR /src

ENV NG_CLI_ANALYTICS "ci"
ARG LIB=/kf/kubeflow/components/crud-web-apps/common/frontend/kubeflow-common-lib
COPY --from=fetch-kubeflow-kubeflow $LIB/package.json ./
COPY --from=fetch-kubeflow-kubeflow $LIB/package-lock.json ./
RUN npm ci

COPY --from=fetch-kubeflow-kubeflow $LIB/projects ./projects
COPY --from=fetch-kubeflow-kubeflow $LIB/angular.json .
COPY --from=fetch-kubeflow-kubeflow $LIB/tsconfig.json .
RUN npm run build

# --- Build the frontend ---
FROM node:16.20.2-bullseye as frontend

COPY ./pkg/ui/v1beta1/frontend ./src
WORKDIR /src

WORKDIR /src

COPY ./pkg/ui/v1beta1/frontend/package.json ./
COPY ./pkg/ui/v1beta1/frontend/package-lock.json ./
COPY ./pkg/ui/v1beta1/frontend/tsconfig.json ./
COPY ./pkg/ui/v1beta1/frontend/tsconfig.app.json ./
COPY ./pkg/ui/v1beta1/frontend/tsconfig.spec.json ./
COPY ./pkg/ui/v1beta1/frontend/angular.json ./
COPY ./pkg/ui/v1beta1/frontend/src ./src

ENV NG_CLI_ANALYTICS "ci"
RUN npm ci
COPY --from=frontend-kubeflow-lib /src/dist/kubeflow/ ./node_modules/kubeflow/

RUN npm run build:prod

# --- Build the backend ---
FROM golang:alpine AS go-build

ARG TARGETARCH

WORKDIR /go/src/github.com/kubeflow/katib

# Download packages.
COPY go.mod .
COPY go.sum .
RUN go mod download -x

# Copy sources.
COPY cmd/ cmd/
COPY pkg/ pkg/

# Build the binary.
RUN CGO_ENABLED=0 GOOS=linux GOARCH=${TARGETARCH} go build -a -o katib-ui  ./cmd/ui/v1beta1

# --- Compose the web app ---
FROM alpine:3.15
WORKDIR /app
COPY --from=go-build /go/src/github.com/kubeflow/katib/katib-ui /app/
COPY --from=frontend /src/dist/static /app/build/static/
ENTRYPOINT ["./katib-ui"]
