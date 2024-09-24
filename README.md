# Modified PointNet++ based on Pytorch

Este repositorio es la implementación de la red PointNet++ (http://papers.nips.cc/paper/7095-pointnet-deep-hierarchical-feature-learning-on-point-sets-in-a-metric-space.pdf) en Pytorch para el Trabajo de Fin de Master - UPV "Desarrollo de un clasificador de modelos óseos con inteligencia artificial mediante PointNet++ y escaneado 3D para el primer banco virtual de Europa" desarrollado dentro del BTELab (Valencia)

En el proyecto del primer banco virtual de tejidos de Europa, en colaboración con el Banco de Células y Tejidos de la Comunidad Valenciana, se busca reducir la desestimación de muestras producida durante la selección de tejidos para implantación. Para ello, se emplea un escaneado 3D mediante luz estructurada, generando un modelo digital y obteniendo una base de datos de las muestras. Así se obtienen 200 muestras, principalmente de tejidos osteotécnicos donde no es necesario ambiente aséptico, facilitados por el Departamento de Anatomía y Embriología Humana de la Universidad de Valencia.

![Resultados de huesos escaneados mediante luz estructurada](images/bones_sle.png)

El conjunto de modelos digitalizados es procesado por una red neuronal basada en PointNet++, con particiones aleatorias del 60% para entrenamiento, 20% para validación y 20% para test. La red tiene diez hiperparámetros, optimizados en función del rendimiento y las métricas de exactitud, kappa de Cohen y coeficiente de correlación de Matthews. 

El modelo final presenta una exactitud de 0.9917 en entrenamiento, 0.8750 en validación y 0.7750 en test. Estos resultados son satisfactorios, de forma que más del 70% de las nuevas muestras serán identificadas correctamente de forma automática, para una futura implementación dentro del proyecto.

! [Matriz de confusión y gráficas para entrenamiento y validación] (images/train_validation.png)
