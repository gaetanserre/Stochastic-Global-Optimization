# Credits https://github.com/centreborelli/ipol-docker-images/

FROM python:3.10

RUN pip install --no-cache-dir pip==23.0.1

COPY .ipol/packages.txt packages.txt
RUN apt-get update && apt-get install -y $(cat packages.txt) && rm -rf /var/lib/apt/lists/*

RUN opam init --disable-sandboxing && eval $(opam env) && opam install ocamlbuild menhir -y

ENV bin /workdir/bin/
RUN mkdir -p $bin
WORKDIR $bin
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

WORKDIR $bin/numpy_parser
RUN eval $(opam env) && make

WORKDIR $bin
RUN chmod -R 777 *