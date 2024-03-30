// LunarLander.cpp
#include <iostream>
#include <tensorflow/c/c_api.h>

class DuelingDoubleDeepQNetwork {
public:
    DuelingDoubleDeepQNetwork(int num_actions, int fc1, int fc2) {
        TF_Status *status = TF_NewStatus();
        TF_Graph *graph = TF_NewGraph();

        // Define placeholders
        TF_Output state = {nullptr, 0};
        TF_OperationDescription *placeholder_desc = TF_NewOperation(graph, "Placeholder", "state");
        TF_SetAttrType(placeholder_desc, "dtype", TF_FLOAT);
        state.oper = TF_FinishOperation(placeholder_desc, status);
        if (TF_GetCode(status) != TF_OK) {
            std::cerr << "Error creating placeholder for state: " << TF_Message(status) << std::endl;
            return;
        }

        // Define layers
        TF_OperationDescription *dense1_desc = TF_NewOperation(graph, "Dense", "dense1");
        TF_SetAttrType(dense1_desc, "T", TF_FLOAT);
        TF_SetAttrInt(dense1_desc, "units", fc1);
        TF_SetAttrString(dense1_desc, "activation", "relu", strlen("relu"));
        TF_Operation *dense1 = TF_FinishOperation(dense1_desc, status);

        TF_OperationDescription *dense2_desc = TF_NewOperation(graph, "Dense", "dense2");
        TF_SetAttrType(dense2_desc, "T", TF_FLOAT);
        TF_SetAttrInt(dense2_desc, "units", fc2);
        TF_SetAttrString(dense2_desc, "activation", "relu", strlen("relu"));
        TF_Operation *dense2 = TF_FinishOperation(dense2_desc, status);

        TF_OperationDescription *V_desc = TF_NewOperation(graph, "Dense", "V");
        TF_SetAttrType(V_desc, "T", TF_FLOAT);
        TF_SetAttrInt(V_desc, "units", 1);
        TF_SetAttrString(V_desc, "activation", nullptr, 0);
        TF_Operation *V = TF_FinishOperation(V_desc, status);

        TF_OperationDescription *A_desc = TF_NewOperation(graph, "Dense", "A");
        TF_SetAttrType(A_desc, "T", TF_FLOAT);
        TF_SetAttrInt(A_desc, "units", num_actions);
        TF_SetAttrString(A_desc, "activation", nullptr, 0);
        TF_Operation *A = TF_FinishOperation(A_desc, status);

        // Build computation graph
        TF_Operation *V_output = TF_GraphOperationByName(graph, "V");
        TF_Output V_output_op = {V_output, 0};

        TF_Operation *A_output = TF_GraphOperationByName(graph, "A");
        TF_Output A_output_op = {A_output, 0};

        // Define Q calculation
        TF_OperationDescription* sum_op_desc = TF_NewOperation(graph, "Sum", "sum");
        TF_OperationDescription* count_op_desc = TF_NewOperation(graph, "Size", "count");

        // Проверяем успешность создания операций sum_op и count_op
        TF_Operation* sum_op = TF_FinishOperation(sum_op_desc, status);
        TF_Operation* count_op = TF_FinishOperation(count_op_desc, status);

        if (sum_op == nullptr || count_op == nullptr) {
            std::cerr << "Ошибка при создании sum_op или count_op: " << TF_Message(status) << std::endl;
            // return;
        }

        // Добавляем входной тензор для операции Size
        TF_AddInput(count_op_desc, {state.oper, 0}); // Используем операцию state в качестве входа для операции Size

        // Затем, делим сумму на количество, чтобы получить среднее значение
        TF_OperationDescription* div_op_desc = TF_NewOperation(graph, "Div", "avg_A");
        TF_AddInput(div_op_desc, {sum_op, 0});
        TF_AddInput(div_op_desc, {count_op, 0});
        TF_FinishOperation(div_op_desc, status);

        if (TF_GetCode(status) != TF_OK) {
            std::cerr << "Error building computation graph: " << TF_Message(status) << std::endl;
            return;
        }

        // Save graph and session
        TF_Buffer *graph_def = TF_NewBuffer();
        TF_GraphToGraphDef(graph, graph_def, status);
        if (TF_GetCode(status) != TF_OK) {
            std::cerr << "Error converting graph to GraphDef: " << TF_Message(status) << std::endl;
            return;
        }

        graph_def_ = graph_def;
        graph_ = graph;

        TF_DeleteStatus(status);
    }

    ~DuelingDoubleDeepQNetwork() {
        TF_DeleteBuffer(graph_def_);
        TF_DeleteGraph(graph_);
    }

private:
    TF_Buffer *graph_def_;
    TF_Graph *graph_;
};

int main() {
    int num_actions = 10;
    int fc1 = 128;
    int fc2 = 64;

    DuelingDoubleDeepQNetwork dddqn(num_actions, fc1, fc2);

    return 0;
}
