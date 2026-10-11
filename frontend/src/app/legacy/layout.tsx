import { ModelProvider } from "@/lib/model-context";
import { DatasetProvider } from "@/lib/dataset-context";

export default function LegacyLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <ModelProvider>
      <DatasetProvider>{children}</DatasetProvider>
    </ModelProvider>
  );
}
