FROM jupyter/scipy-notebook:44fe185f9495
# pandas、numpy、scipy、 scikit-image、matplotlibなど一通り入っているjupyter/scipy-notebookを使う。
# また、常にpullするimageを固定したいので、一意のtagをつける。
USER jovyan
COPY requirements.yml /home/jovyan/
RUN conda env update --name base --file /home/jovyan/requirements.yml --prune && \
    jupyter nbextension enable highlight_selected_word/main && \
    conda clean -i -t -y